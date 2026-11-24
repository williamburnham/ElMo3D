"""
ROM extraction — per-trial joint range of motion for every ElMo3D sub-dataset,
averaged to one row per participant in participant_rom.csv.

This is step 1 and the only step that needs the dataset. Step 2 (statistics)
is ROM_ANOVA.py; the figure is plot_distributions.py.

Angles come from the AMASS release (`data/ElMo3D_data/motion/SMPLH_AMASS`), not from the 263-dim
vectors. In SMPL the rotation stored at a joint is that joint's own rotation
relative to its parent — pose[1] is the thigh w.r.t. the pelvis — so the Euler
components are anatomical by construction.

HumanML3D's `rot_data` was tried first and rejected: its local rotations are
relative to the parent's chained qbetween frame rather than an anatomical joint
coordinate system. The knee happened to agree with the literature (62-72 deg vs
~60) because its parent segment points downward, but the hip came out at 75-111
deg against a literature 40-45, and produced a spurious 28 deg left/right
asymmetry in a healthy subject. It also carries no twist, since qbetween gives
the minimal rotation.
"""

import os
import os.path as osp
import csv
from glob import glob

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

FPS = 20
_RELEASE = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..', '..'))
AMASS_DIR = osp.join(_RELEASE, 'data', 'ElMo3D_data', 'motion', 'SMPLH_AMASS')
TRIALS_CSV = osp.join(_RELEASE, 'data', 'ElMo3D_data', 'trials.csv')

# SMPL-H pose layout: 156 = 3 root + 21*3 body + 15*3 left hand + 15*3 right hand
BODY0, N_BODY = 3, 21

JOINT_NAMES = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee', 'spine2',
    'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot', 'neck', 'L_collar',
    'R_collar', 'head', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist',
]

# In SMPL the index is the joint itself — no off-by-one. pose[1] rotates the
# thigh, pose[4] the shank, pose[7] the foot.
ANGLE_AT = {
    'L_hip': 1,  'R_hip': 2,
    'L_knee': 4, 'R_knee': 5,
    'L_ankle': 7, 'R_ankle': 8,
    'lumbar': 3, 'thoracic': 6, 'upper_thoracic': 9,
    'neck': 12,
    'L_shoulder': 16, 'R_shoulder': 17,
    'L_elbow': 18,    'R_elbow': 19,
}

COMPONENTS = ('flexion', 'abduction', 'rotation')

GROUPS = ['138H', '50ST', '24MP', '1MPS', 'G120', '9WRT', '10KH', '10KS']

# Normal walking only. G120 also holds stairs, slopes and sit-to-stand; 9WRT
# holds running, jumping, squats and lunges; MovePort holds deliberately
# altered gaits (dragging, leg-high) and standing. Comparing joint ROM across
# sites is meaningless if the groups are doing different things.
# Treadmill vs overground does not matter here: these are joint angles, so belt
# motion does not enter.
# After the rename the action is a field, so gait is simply the two gait codes.
GAIT_ACTIONS = {'GaitGr', 'GaitTm'}


# MovePort imposes three treadmill speeds. Pooling them produces a cohort mean
# that reflects how many trials the protocol ran at each speed rather than how
# the cohort walks: within 24MP alone, low -> high moves ROM by 10-11 deg, about
# twice the size of the between-cohort differences being tested. Each speed is
# therefore its own level.
# ground_gait is kept separate too — per MovePort's readme those trials exist
# only for the participant with spastic paraplegia (Subj25).
# The rename merged MovePort's three belt speeds into one action (GaitTm), so
# the speed is no longer readable from the filename. It survives only in the
# repetition number (high 1-2, low 3-4, medium 5-6 per participant) and in
# trials.csv (imposed_speed). Rebuild the split from there if the speed levels are wanted.
SPEED_SPLIT = {}


# MovePort's three imposed treadmill speeds. The rename merged them into one
# action (GaitTm), so the speed is no longer in the filename -- it survives only
# in trials.csv (imposed_speed, from the original trial stems). The map below is
# exact (163/163 trials, no conflicts) and reproduces MovePort's published
# 63 low / 51 medium / 49 high split. Do NOT infer speed from the repetition
# number: reps 1-2 are reliably 'high' but 3-14 mix low and medium.
#
# Only 'high' is kept. MovePort's readme describes it as 3-4 km/h "comfortable
# brisk walking", the closest of the three to the self-selected overground gait
# every other cohort walked, so it is the only level comparable to them.
# Pooling all three instead makes 24MP's cohort mean a function of how many
# trials the protocol ran at each speed: within 24MP, low -> high moves ROM by
# a mean of 6.7 deg and up to 21.6 deg (L_ankle_flexion_ROM), several times the
# between-cohort differences under test.
KEEP_SPEED = {'24MP': {'high'}}


def _load_speed_map():
    """-> {trial_stem: speed} for cohorts with imposed speeds (trials.csv)."""
    out = {}
    with open(TRIALS_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['subdataset'] in KEEP_SPEED and r['imposed_speed']:
                out[r['trial']] = r['imposed_speed']
    return out


SPEED_MAP = _load_speed_map()


def speed_ok(group, stem):
    """False for a trial at an imposed speed we are not analysing."""
    keep = KEEP_SPEED.get(group)
    if not keep:
        return True
    sp = SPEED_MAP.get(stem)
    if sp is None:
        return True          # not an imposed-speed trial (e.g. overground)
    return sp in keep


def group_label(group, stem):
    """Analysis group for a trial, splitting imposed-speed conditions apart."""
    for pat, lab in SPEED_SPLIT.get(group, ()):
        if pat in stem:
            return lab
    return group


def subject_of(stem):
    """'50ST-Strk-GaitGr-0008-0300' -> '50ST-0008'."""
    f = stem.split('-')
    return f'{f[0]}-{f[3]}'


def class_of(stem):
    """Class is the second field of the name."""
    return {'Hlth': 'Healthy', 'Strk': 'Stroke',
            'SpPa': 'Spastic Paraplegia'}.get(stem.split('-')[1], 'Unknown')


def is_gait(group, stem):
    f = stem.split('-')
    return (len(f) >= 3 and f[2] in GAIT_ACTIONS
            and speed_ok(group, stem))

# Upper-body angles for these groups come from MoSh's pose prior, not markers.
UNMEASURED = {
    '9WRT': {'neck', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow'},
    '10KH': {'lumbar', 'thoracic', 'upper_thoracic', 'neck',
             'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow'},
    '10KS': {'lumbar', 'thoracic', 'upper_thoracic', 'neck',
             'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow'},
}



def rest_bone_axes():
    """Dominant rest-pose axis of the bone each joint rotates: {joint: 'X'|'Y'|'Z'}.

    The Euler sequence puts the bone's own long axis last, leaving twist as the
    final component. Computed from the neutral SMPL-H model's rest pose
    (J_regressor @ v_template, child joint minus joint); fixed here because the
    SMPL-H model is licensed separately and not distributed with ElMo3D.
    """
    return {'L_hip': 'Y', 'R_hip': 'Y', 'L_knee': 'Y', 'R_knee': 'Y',
            'L_ankle': 'Z', 'R_ankle': 'Z', 'lumbar': 'Y', 'thoracic': 'Y',
            'upper_thoracic': 'Y', 'neck': 'Y', 'L_shoulder': 'X',
            'R_shoulder': 'X', 'L_elbow': 'X', 'R_elbow': 'X'}


# Sequence puts the bone axis last, so component 3 is twist about the bone.
SEQ = {'X': 'ZYX', 'Y': 'XZY', 'Z': 'XYZ'}


def joint_angles(poses):
    """{joint: [T, 3] degrees} as (flexion, abduction, rotation).

    `poses` is the AMASS [T, 156] fullpose array.
    """
    T = len(poses)
    body = poses[:, BODY0:BODY0 + N_BODY * 3].reshape(T, N_BODY, 3)
    axes = rest_bone_axes()
    out = {}
    for name, j in ANGLE_AT.items():
        aa = body[:, j - 1]                       # body block starts at joint 1
        eul = Rotation.from_rotvec(aa).as_euler(SEQ[axes[name]], degrees=True)
        out[name] = eul
    return out


def calculate_min_max_rom(poses, robust=True):
    """
        takes a sample, calculates the minimum, maximum, and ROM of every joint

    One triple per joint per component (flexion / abduction / rotation).
    `robust=True` uses the 5th-95th percentile rather than the raw extremes: a
    single glitched frame otherwise sets both ends and ROM measures the worst
    artefact in the clip instead of the movement.

    Note min and max are relative to the SMPL rest pose, not to clinical
    neutral standing. ROM is unaffected — a constant offset cancels in max-min —
    but the min/max columns are only comparable between clips, not to published
    normative angles.
    """
    row = {}
    for name, eul in joint_angles(poses).items():
        for k, comp in enumerate(COMPONENTS):
            a = eul[:, k]
            if robust:
                lo, hi = np.percentile(a, 5), np.percentile(a, 95)
            else:
                lo, hi = a.min(), a.max()
            row[f'{name}_{comp}_min'] = float(lo)
            row[f'{name}_{comp}_max'] = float(hi)
            row[f'{name}_{comp}_ROM'] = float(hi - lo)
    return row


def calculate_dataset(group, gait_only=True, out_dir='.', robust=True):
    """
        runs calculate_min_max_rom() for every sample in a dataset and saves them all to a csv
    """
    # Recorded trials only: the treadmill 'translated' copies are the same
    # trials re-expressed, and joint angles are identical in both frames.
    files = sorted(f for f in glob(osp.join(AMASS_DIR, '**', group + '-*.npz'),
                                   recursive=True)
                   if osp.sep + 'translated' + osp.sep not in f)
    rows, skipped = [], 0
    for f in files:
        stem = osp.splitext(osp.basename(f))[0]
        if gait_only and not is_gait(group, stem):
            skipped += 1
            continue
        d = np.load(f, allow_pickle=True)
        poses = d['poses']
        row = calculate_min_max_rom(poses, robust=robust)
        # Blank the joints this group never measured, so they cannot contribute
        # a difference that is really MoSh's prior talking.
        for j in UNMEASURED.get(group, ()):
            for c in COMPONENTS:
                for k in ('min', 'max', 'ROM'):
                    row[f'{j}_{c}_{k}'] = np.nan
        row.update(trial=stem, group=group_label(group, stem),
                   source_group=group, subject=subject_of(stem),
                   clinical_class=class_of(stem), n_frames=len(poses),
                   framerate=float(d['mocap_framerate']))
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, f'rom_{group}.csv')
    df.to_csv(path, index=False)
    bits = ''
    if group in SPEED_SPLIT and len(df):
        bits = '  [' + ', '.join(f'{k}:{v}' for k, v in
                                 df.group.value_counts().sort_index().items()) + ']'
    print(f'  {group:<6} {len(df):>5} trials  ({skipped} non-gait skipped)'
          f'  {df["subject"].nunique():>4} subjects{bits}')
    return df


def participant_table(df, out_dir='.'):
    """One row per participant: every ROM feature, averaged over their trials.

    Columns are ordered so the trustworthy measures come first — lower-limb
    flexion validates against published gait norms, whereas the abduction and
    rotation components, and the spine and arm joints, are exploratory.
    Blank cells are joints the group never measured (NaN by design, not error).
    """
    rom = [c for c in df.columns if c.endswith('_ROM')]
    limb = ['L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']
    first = [f'{j}_flexion_ROM' for j in limb]
    rest = [c for c in rom if c not in first]
    order = first + sorted(rest)

    g = df.groupby(['group', 'subject'])
    out = g[order].mean()   # full precision: ROM_ANOVA.py tests these values
    out.insert(0, 'n_trials', g.size())
    out.insert(1, 'clinical_class', g['clinical_class'].first())
    out.insert(2, 'mean_trial_frames', g['n_frames'].mean().round(0).astype(int))
    out = out.reset_index().sort_values(['group', 'subject'])
    path = osp.join(out_dir, 'participant_rom.csv')
    out.to_csv(path, index=False)
    print(f'wrote {osp.basename(path)}: {len(out)} participants x {len(order)} ROM features')
    return out



def main():
    """Per-trial ROM for every group -> participant_rom.csv. No trials or
    values are excluded. ROM_ANOVA.py and plot_distributions.py run from
    participant_rom.csv, so they do not need the dataset."""
    here = osp.dirname(osp.abspath(__file__))
    print('per-trial ROM, normal gait only:')
    df = pd.concat([calculate_dataset(g, out_dir=here) for g in GROUPS],
                   ignore_index=True)
    df.to_csv(osp.join(here, 'rom_all.csv'), index=False)
    print(f'\ntotal {len(df)} trials, {df["subject"].nunique()} subjects, '
          f'{df["group"].nunique()} groups\n')
    participant_table(df, out_dir=here)
    return df


if __name__ == '__main__':
    main()
