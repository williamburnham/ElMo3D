# ElMo3D: A Dataset of Healthy and Post-Stroke Motion Capture and Electromyography for Machine Learning

ElMo3D pairs full-body motion with surface EMG from multiple sub-datasets of healthy,
stroke and spastic-paraplegia participants. Everything is stored once, in its most
complete form, under `data/ElMo3D_data/`.
`build_dataset.py` then assembles whichever version of the dataset you need into
`dataset_build/`.


## 1. Quick start

```bash
cd ElMo3D
python build_dataset.py --list                       # what is available
python build_dataset.py --motion 263dim_HumanML3D_normalisation \
                        --emg processed --trials all_translated
python build_dataset.py --config config_example.json  # options from a file
```

The second command writes a new folder `dataset_build/<date>_<time>/` holding the
motion files, the EMG cut to match them, and `build_info.json`.
Requires Python 3 and numpy.


## 2. Folder layout

```
ElMo3D/
  build_dataset.py           builds a dataset version (section 5)
  config_example.json        example build options for --config
  README.md                  this file
  dataset_build/             builds are written here
  validation/ROM_ANOVA/      hip/knee ROM: Welch's ANOVA / t-test across cohorts
  visualisation/             clip viewer: skeleton and EMG, video export (section 9)
  data/
    ElMo3D_data/                      the dataset (not in the git repository)
      motion/
        263dim_HumanML3D_normalisation/{new_joints,new_joint_vecs}/<category>/
        263dim_original_proportions/{new_joints,new_joint_vecs}/<category>/
        SMPLH_AMASS/<category>/
      EMG/
        processed/<emg category>/      one .npy per trial
        raw/<emg category>/            one .npz per trial
        normalised/<emg category>/     one .npy per trial
        emg_channels.json              channel order and units
      index_2.csv                      clip windows of the original trials
      index_2_translated.csv           clip windows of the translated trials
      trials.csv                       one row per trial (section 6)
      participant_information.csv      sex, age, weight, lesion location, ...
```

The git repository holds the code and documentation only. The data (17 GB) is
distributed separately: put its contents in `data/ElMo3D_data/`, so that
`data/ElMo3D_data/trials.csv` exists, before running `build_dataset.py`.
Download the data from: [ElMo3D figshare](https://doi.org/10.6084/m9.figshare.35107477).

## 3. Naming

Every file is named

```
{subdataset}-{class}-{action}-{participant}-{repetition}{clip}
```

e.g. `1MPS-SpPa-GaitGr-0025-0100.npy`: subdataset 1MPS, class SpPa (spastic
paraplegia), action GaitGr, participant 0025, repetition 01, clip 00.

| | |
|---|---|
| class | Hlth healthy, Strk stroke, SpPa spastic paraplegia |
| clip | trials longer than 196 frames (9.8 s) are cut into centred, non-overlapping 196-frame clips: 00, 01, 02, ... |
| `M` prefix | `M1MPS-SpPa-GaitGr-0025-0100.npy` is the left/right mirror of that clip (263-dim representations only; EMG is mirrored by swapping left and right channels) |

Trial-level files (SMPLH_AMASS, EMG) have no clip digits:
`1MPS-SpPa-GaitGr-0025-01`. Inside a build, the EMG file always carries exactly
the name of the motion file it belongs to.


## 4. What each representation is

**SMPLH_AMASS** (`.npz`, one per trial, mocap frame rate)
SMPL-H body fitted to the markers by MoSh++, in AMASS
format: `poses` [T,156] axis-angle, `trans` [T,3] metres (Z up), `betas` [16],
`gender`, `mocap_framerate`. Translated treadmill trials also carry
`treadmill_belt_velocity` [3] (m/s, AMASS frame) and `trim_start_frame` /
`trim_end_frame` (mocap frames of the untrimmed trial).
Every file also carries `measured_joints` [52] (bool, SMPL-H joint order, as
in `poses`).
The SMPL-H model itself is licensed separately (mano.is.tue.mpg.de).

**263dim_HumanML3D_normalisation** (`.npy`, one per clip, 20 fps)
The HumanML3D motion representation. `new_joint_vecs` [T-1, 263]: root
angular velocity, root linear velocity, root height, joint positions
relative to the root, 6D joint rotations, joint velocities and four foot
contacts. `new_joints` [T-1, 22, 3]: the joint positions recovered from the
vector. Every subject is retargeted onto the skeleton of HumanML3D clip
000021, as HumanML3D does, so body proportions are the same for everyone.

**263dim_original_proportions** (`.npy`, one per clip, 20 fps)
The same representation, but each subject keeps their own skeleton: the
median bone lengths over all their trials, from the fitted SMPL-H body.
Mirrored clips use the mirrored skeleton.

**EMG/processed** (`.npy`, [20 channels, N], 100 Hz, microvolts)
Band-pass 20-450 Hz, full-wave rectified, critically damped low-pass
envelope, resampled to 100 Hz.

**EMG/normalised** (`.npy`, [20 channels, N], 100 Hz, unitless)
EMG/processed with bilateral peak normalisation.
Per file and per muscle, the left and right channels are divided by the larger
of their two peaks within the middle 90% of the file, so the louder side peaks
at 1 and the quieter side keeps its relative amplitude. The first and last 5%
are excluded from the peak because of filter edge artefacts. They are still
scaled, so values there can fall outside [0, 1]. One-sided muscles use their
own peak; unrecorded channels stay zero.

**EMG/raw** (`.npz`: `emg` [20, N] float32 microvolts, `fs`, `labels`)
The C3D analog EMG converted to microvolts, unfiltered, at the native
rate (1000, 2000 or 2048 Hz depending on the sub-dataset).

EMG channels (`emg_channels.json`): RERS LERS RRF LRF RVL LVL RBF LBF RST LST
RTA LTA RGAS LGAS RLG LLG RSOL LSOL RIL LIL (erector spinae, rectus femoris,
vastus lateralis, biceps femoris, semitendinosus, tibialis anterior,
gastrocnemius medial head, lateral gastrocnemius, soleus, iliopsoas; R/L).
A channel a sub-dataset did not record is all zeros.


## 5. build_dataset.py

```bash
python build_dataset.py --motion <base> --trials <type> [options]
```

| Option | Values |
|---|---|
| `--motion` | `263dim_original_proportions`<br>`263dim_HumanML3D_normalisation`<br>`SMPLH_AMASS` |
| `--emg` | `processed` (default) \| `raw` \| `none` |
| `--trials` | `overground_gait_only`<br>`original_treadmill_gait_only`<br>`translated_treadmill_gait_only`<br>`overground_and_original_treadmill_gait`<br>`overground_and_translated_treadmill_gait`<br>`all_original`<br>`all_translated` |
| `--subdatasets` | `all` (default) or a list: `--subdatasets 138H 50ST G120` |
| `--exclude` | leave sub-datasets out: `--exclude 1MPS` |
| `--name` | output folder name (default: current date and time, e.g. `2026-11-03_14-05-12`) |
| `--config` | read the options above from a JSON file (see below) |
| `--list` | print the sub-datasets, trial counts and options, then exit |

"original" treadmill trials are as recorded on the belt; "translated" ones are
the same trials re-expressed as overground walking (section 7). The two never
appear in the same build. `all_original` and `all_translated` include every
overground trial plus the treadmill trials in that version.

### Config files

Edit the options in a JSON file and pass it with `--config`.

`config_example.json`:

```json
{
  "_comment": "Options for build_dataset.py --config. Keys starting with _ are ignored. See README.md, section 5.",
  "motion": "263dim_HumanML3D_normalisation",
  "emg": "processed",
  "trials": "overground_and_translated_treadmill_gait",
  "subdatasets": "all",
  "exclude": ["1MPS"],
  "name": "walking_translated_no_1MPS"
}
```

```bash
python build_dataset.py --config config_example.json
python build_dataset.py --config config_example.json --emg raw --name walking_raw_emg
```

Options given on the command line override the file, as in the second line.
`subdatasets` and `exclude` take a list or a single name; `name` may be `null`
for the date and time. `build_info.json` records the path of the config file and
the options actually used.

Output, `dataset_build/<name>/`:

```
263-dim bases:
  motion/new_joint_vecs/*.npy     the 263-dim vectors
  motion/new_joints/*.npy         the matching joint positions
  Mean.npy, Std.npy               computed over this build (see below)
SMPLH_AMASS:
  motion/*.npz                    one per trial
with EMG:
  emg/*.npy (processed) or emg/*.npz (raw), same names as the motion files
  emg_Mean.npy, emg_Std.npy       per channel, over this build
always:
  build_info.json                 options, counts per sub-dataset, and the
                                  motion files that have no EMG
```

Mean/Std follow HumanML3D: per-dimension mean, one standard deviation shared
within each feature block. Motion files without EMG are kept and listed
under `unpaired_motion` in `build_info.json`.

### Examples

Healthy and stroke level walking, overground only, HumanML3D skeleton:

```bash
python build_dataset.py --motion 263dim_HumanML3D_normalisation \
    --trials overground_gait_only --name walking_overground
```

All walking with treadmill trials as overground, subject skeletons, no
spastic-paraplegia participant:

```bash
python build_dataset.py --motion 263dim_original_proportions \
    --trials overground_and_translated_treadmill_gait --exclude 1MPS
```

Everything, as recorded, with raw EMG, for two sub-datasets:

```bash
python build_dataset.py --motion SMPLH_AMASS --emg raw \
    --trials all_original --subdatasets 24MP 9WRT
```

Motion only:

```bash
python build_dataset.py --motion 263dim_HumanML3D_normalisation \
    --emg none --trials all_translated
```


## 6. trials.csv and the indices

`trials.csv` has one row per trial:

| Column | |
|---|---|
| `trial`, `subdataset`, `clinical_class`, `action`, `participant`, `repetition`, `category` | |
| `mocap_fps` | capture rate of the motion |
| `stage3_step` | redundant; was used during data harmonising |
| `resampled_to_20fps` | 1 where the motion was interpolated onto the 20 fps grid instead (9WRT, captured at 128 Hz) |
| `motion_fps` | frame rate of the 263-dim files: 20 for every trial |
| `n_mocap_frames` | number of frames of motion capture |
| `imposed_speed` | 24MP treadmill gait: low / medium / high |
| `translated`, `translation_status`, `belt_speed_mps`, `trim_start_frame`, `trim_end_frame` | treadmill translation details |
| `emg_processed` | 1 if the trial has EMG |
| `emg_raw_fs` | native EMG rate |
| `unmeasured_joints` | SMPL-H body joints with no marker data; blank if all were measured |

`index_2.csv` and `index_2_translated.csv` give each clip's window inside its trial
(`trial`, `start_frame`, `end_frame`, `new_name`), in 263-dim frames of the untrimmed
trial. `build_dataset.py` uses them to cut EMG.


## 7. Things to know

### Sub-datasets

Size. Clips are the 196-frame clips of the original (untranslated) trials,
without mirrors. Duration is the total capture time. Translated is the number of
treadmill trials that also have a translated version.

| Sub-dataset | Source | Class | Participants | Trials | Clips | Duration (min) | Trials with EMG | Translated |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 138H | Plug-in Gait | Healthy | 138 | 449 | 449 | 22.6 | 340 | – |
| 50ST | Plug-in Gait | Stroke | 50 | 150 | 153 | 16.5 | 150 | – |
| 24MP | MovePort | Healthy | 24 | 370 | 1,475 | 264.8 | 370 | 276 |
| 1MPS | MovePort | Spastic paraplegia | 1 | 22 | 31 | 4.0 | 22 | – |
| G120 | Gait120 | Healthy | 120 | 4,126 | 4,126 | 149.7 | 4,126 | – |
| 9WRT | split-belt treadmill | Healthy | 9 | 117 | 637 | 110.4 | 117 | 81 |
| 10KH | Knee-Flex | Healthy | 10 | 50 | 50 | 2.8 | 50 | – |
| 10KS | Knee-Flex | Stroke | 10 | 50 | 63 | 11.7 | 50 | – |
| **Total** | | | **362** | **5,334** | **6,984** | **582.5** | **5,225** | **357** |

Recording and activities. Mocap and EMG rates are the native capture rates
(every 263-dim file is 20 fps, processed EMG 100 Hz). EMG muscles are listed
without the R/L prefix (channels in section 4).

| Sub-dataset | Setting | Activities (action codes, trials) | Mocap (Hz) | EMG (Hz) | EMG muscles | Unmeasured joints |
|---|---|---|---:|---:|---|---|
| 138H | overground | level walking (GaitGr 449) | 100 | 1000 | ERS RF VL BF ST TA GAS, both legs | – |
| 50ST | overground | level walking (GaitGr 150) | 100 | 1000 | ERS RF VL BF ST TA GAS, both legs (LSOL in some trials) | – |
| 24MP | treadmill + overground | treadmill walking at low/medium/high speed (GaitTm 163); dragging (Draggi 59), high knees (LegHig 56), standing balance (StndSt 26, StndBk 24, StndSq 22, StndFw 20) | 100 | 2000 | RF VL ST TA GAS LG SOL IL, both legs | – |
| 1MPS | overground | level walking (GaitGr 13); standing balance (StndBk 3, StndSt 3, StndFw 2, StndSq 1) | 100 | 2000 | RF VL ST TA GAS LG SOL IL, both legs | – |
| G120 | overground | level walking (GaitGr 600), slope ascent/descent (SlopeA 600, SlopeD 600), stair ascent/descent (StairA 599, StairD 600), sit-to-stand (SitStd 558), stand-to-sit (StdSit 569) | 100 | 2000 | RF VL BF ST TA GAS LG SOL, right leg only | – |
| 9WRT | split-belt treadmill + overground | treadmill walking (GaitTm 54), running (Runnin 27); jumping, landing, lunging, squatting (9 each) | 128 | 2048 | RF VL BF ST TA GAS LG SOL, right leg only | neck, head, collars, arms |
| 10KH | overground | level walking (GaitGr 50) | 200 | 1000 | RF BF TA GAS, both legs | spine, neck, head, collars, arms |
| 10KS | overground | level walking (GaitGr 50) | 200 | 1000 | RF BF TA GAS, both legs | spine, neck, head, collars, arms |

Participants (from `participant_information.csv`; blank where the source
provides no demographics). Age and weight are mean ± SD.

| Sub-dataset | Sex (M/F) | Age (years) | Weight (kg) | Clinical details |
|---|---|---|---|---|
| 138H | 65/73 | 51.2 ± 19.6 (21–86) | 74.4 ± 15.1 | – |
| 50ST | 34/16 | 64.2 ± 14.0 (19–85) | 72.0 ± 13.6 | 39 ischaemic, 11 haemorrhagic; lesion 32 right, 18 left; 53 ± 29 days post-stroke (14–139) |
| 24MP | | | | – |
| 1MPS | | | | spastic paraplegia |
| G120 | 120/0 | 35.4 ± 12.9 (20–59) | 74.2 ± 12.5 | – |
| 9WRT | 5/4 | 25.4 ± 2.7 (23–30) | 68.1 ± 9.0 | – |
| 10KH | | | | – |
| 10KS | | | | stroke |

### Treadmill translation

Translated trials remove the belt velocity from the SMPL-H root translation,
so they read as overground walking. Every translated 263-dim clip is computed
from the translated SMPL-H. The belt velocity is estimated from the stance feet,
one velocity per trial. MovePort starts the belt with the participant on
it, so trials are trimmed to the stretch where the belt runs at steady
speed (trimmed at the start in 183 trials, median 3.5 s), and a translated
clip may therefore cover a different window from the original clip of the
same name. Two short trials with no steady stretch have no translated
version. Translation changes the reference frame, not the gait: treadmill
walking still differs biomechanically from overground walking.

### EMG amplitude

All EMG is in microvolts, but no MVC normalisation exists, so amplitudes
also reflect each lab's electrodes and amplifiers. Compare amplitudes
between sub-datasets with care.

### Frame rate

Every 263-dim file is exactly 20 fps. Captures at 100 or 200 Hz keep every
5th or 10th frame; 9WRT, captured at 128 Hz, is computed at the full rate
and interpolated onto the 20 fps grid. SMPLH_AMASS keeps each capture's own
rate (`mocap_framerate`).

### Motion without EMG

Some trials have no EMG recording; their motion is still included.


## 8. Visualisation

`visualisation/visualise_tool.py` plays a clip's skeleton next to its EMG, and
saves what is on screen as a video (`.mp4`, `.gif`) or as a still figure of
overlaid frames. Needs numpy, matplotlib, tkinter, and ffmpeg.

```bash
cd visualisation
python visualise_tool.py                                  # open the viewer, add files
python visualise_tool.py 138H-Hlth-GaitGr-0001-0100       # one clip by name
python visualise_tool.py 24MP-Hlth-GaitTm-0013-05 --source hml3d-tt   # all clips of a trial
python visualise_tool.py 138H-Hlth-GaitGr-0001-01 --source amass      # a whole SMPL-H trial
python visualise_tool.py 138H-Hlth-GaitGr-0001-0100 --save clip.mp4   # no window
python visualise_tool.py 138H-Hlth-GaitGr-0001-0100 --save-figure clip.png
```

**Add files** Files can be picked from `new_joints` or `new_joint_vecs` (the matching joint
positions are shown), from `SMPLH_AMASS`, and also from a build in
`dataset_build/`.

| `--source` | joints shown when a clip is named rather than picked |
|---|---|
| `hml3d` (default) | 263dim_HumanML3D_normalisation |
| `hml3d-tt` | the same, with treadmill trials translated |
| `subject` | 263dim_original_proportions |
| `subject-tt` | the same, with treadmill trials translated |
| `amass` | SMPLH_AMASS, one whole trial per file |
| `amass-tt` | the same, with treadmill trials translated |

Overground clips are the same in the original and translated sources.

Unmeasured joints are not drawn (in SMPLH_AMASS files, those marked False in
`measured_joints`), and unrecorded EMG channels are greyed out.
`--emg-scale muscle` scales each left/right pair together, so
the two sides can be compared; `--global-frame` fixes the camera so the subject
is seen travelling. `python visualise_tool.py --help` lists every option.

`--body` (or **SMPL body** in the window) overlays the neutral SMPL-H body
fitted to the joints. It needs the SMPL-H body model, as do exact skeletons for
SMPLH_AMASS files (see below).

### Installing the SMPL-H body model (optional)

The SMPL-H model is licensed separately and is not part of ElMo3D. Everything
except `--body` and exact SMPLH_AMASS skeletons works without it.

1. Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and accept the
   licence.
2. On the **Download** page, download **Extended SMPL+H model** (the version
   used by AMASS), `smplh.tar.xz`.
3. Unpack it into `visualisation/body_models/smplh/`:

   ```bash
   mkdir -p visualisation/body_models/smplh
   tar -xf smplh.tar.xz -C visualisation/body_models/smplh
   ```

4. Check that these three files exist in the correct locations:

   ```
   visualisation/body_models/smplh/neutral/model.npz
   visualisation/body_models/smplh/male/model.npz
   visualisation/body_models/smplh/female/model.npz
   ```


## 9. Code Statement

  Parts of this codebase were developed with assistance from Claude (Anthropic).
  All code was reviewed, tested and validated by the author.


## 10. Licensing and Attribution

  - **Code** is released under the [MIT License](LICENSE). Please keep the copyright notice in any copies or substantial portions of the code.
  - **Dataset** (`data/`) is released under [CC BY 4.0](data/LICENSE). If you share or adapt the data, you must:
    - credit the authors by citing the original papers,
    - link to the licence: https://creativecommons.org/licenses/by/4.0/,
    - indicate if you made changes.

  ```bibtex
    @article{van2023full,
        title={A full-body motion capture gait dataset of 138 able-bodied adults across the life span and 50 stroke survivors},
        author={Van Criekinge, Tamaya and Saeys, Wim and Truijen, Steven and Vereeck, Luc and Sloot, Lizeth H and Hallemans, Ann},
        journal={Scientific data},
        volume={10},
        number={1},
        pages={852},
        year={2023},
        publisher={Nature Publishing Group UK London}
    }
    @article{wang2017associations,
        title={Associations between lower-limb muscle activation and knee flexion in post-stroke individuals: A study on the stance-to-swing phases of gait},
        author={Wang, Wei and Li, Ke and Yue, Shouwei and Yin, Cuiping and Wei, Na},
        journal={PLoS One},
        volume={12},
        number={9},
        pages={e0183865},
        year={2017},
        publisher={Public Library of Science San Francisco, CA USA}
    }
    @article{boo2025comprehensive,
        title={Comprehensive human locomotion and electromyography dataset: Gait120},
        author={Boo, Junyo and Seo, Dongwook and Kim, Minseung and Koo, Seungbum},
        journal={Scientific data},
        volume={12},
        number={1},
        pages={1023},
        year={2025},
        publisher={Nature Publishing Group UK London}
    }
    @article{jiang2024moveport,
        title={MovePort: Multimodal dataset of EMG, IMU, MoCap, and insole pressure for analyzing abnormal movements and postures in rehabilitation training},
        author={Jiang, Xinyu and Li, Jianfeng and Zhu, Zhuozhuang and Liu, Xiangyu and Yuan, Yangyang and Chou, ChihHong and Yan, Shengjie and Dai, Chenyun and Jia, Fumin},
        journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
        volume={32},
        pages={2633--2643},
        year={2024},
          publisher={IEEE}
    }
    @article{wang2023wearable,
        title={A wearable real-time kinetic measurement sensor setup for human locomotion},
        author={Wang, Huawei and Basu, Akash and Durandau, Guillaume and Sartori, Massimo},
        journal={Wearable technologies},
        volume={4},
        pages={e11},
        year={2023},
        publisher={Cambridge University Press}
    }
  ```