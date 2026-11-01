# Webcam Guitar Chord Recognizer — Project Spec

Nov 1, 2026 · @Vatsal

## How to use this doc with Claude Code

Save this doc as SPEC.md in an empty project folder, open Claude Code there, and start with the prompt below. Build one milestone per session and test each with your own guitar before moving on.

> Read SPEC.md. We are building this project milestone by milestone. Start with Milestone 1 only. Before writing code, summarize your plan and list any questions. After each milestone, tell me exactly how to run and test it, then stop and wait for me.

Rules for Claude Code while building:

- Follow the repository structure and tech stack below unless there is a clear reason not to; explain any change first.
- Keep every script runnable from the command line with --help output.
- Never commit datasets, model weights, or recorded video to git; add them to .gitignore.
- Write a short README section for each milestone as it is finished.

## Overview and goals

Build a Python app that watches a webcam, finds the guitar's fretboard, and names the chord being fretted in real time, with no audio and no extra hardware. It is a personal rebuild of a published student project, adapted to run on a normal laptop.

Goals:

- Recognize 5 open chords (C, D, Em, F, G) from a live webcam at 15 frames per second or better on a laptop.
- Reach at least 85% accuracy on a held-out test set recorded by me with my own acoustic guitar.
- Show the chord name, a confidence score, and the fretboard box on screen, and say "uncertain" instead of guessing when the model is unsure.
- Make it easy to add new chords by recording a few minutes of video and retraining.

Out of scope for the first version: audio analysis, barre chords beyond F, left-handed players, and a web or mobile front end.

## Reference project and approach

This spec follows the design of Exploring-Guitar-Chord-Recognition-with-Vision-Models, described in their write-up *Can a Camera Read Your Chord?*. They report 88.07% test accuracy on 5 chords using about 1,600 fretboard images. Read their repo for ideas, but write this project's code fresh.

Their pipeline, which we copy:

1. Detect the fretboard in each frame with a small object detector (they trained RF-DETR Nano on 714 hand-labeled images).
2. Crop the frame to the fretboard box.
3. Preprocess: subtract each image's own mean pixel value ("sample-wise centering") instead of ImageNet normalization, then resize to 229×229.
4. Classify the crop with an ImageNet-pretrained CNN (Inception-ResNet V2) plus a small new head: global average pooling, two fully connected layers with ReLU and dropout, then a linear layer over the chord classes.
5. Smooth predictions over time with an exponential moving average and reject frames whose prediction entropy is too high.

Lessons from their experiments that Claude Code must respect:

- Do not crop to the hand with MediaPipe. Some chords use the same hand shape at different frets, so a hand crop scored about 50% on 3 classes. The crop must include the fretboard from the nut.
- Augmentation is essential. Without it, test accuracy fell from 88.07% to 73.39% even though validation accuracy looked the same.
- Sample-wise centering matters. Standard normalization dropped test accuracy to 83.94%.
- Use transfer learning, and don't go bigger. Training from scratch scored 68.35%, and the larger ConvNeXt overfit to 56.42%.
- Validation accuracy lies. A 13,000-image dataset with only 3 players taught the model backgrounds and people, not chords. The test set must use different players, rooms, and angles than training.
- Em and G were the weakest classes; G was most often confused with C.

## Tech stack

Everything is Python 3.11 and runs locally; use CUDA or Apple MPS when available and fall back to CPU.

| Part | Choice | Notes |
|---|---|---|
| Video capture and drawing | OpenCV (opencv-python) | Webcam, video files, and on-screen overlay |
| Fretboard detector | RF-DETR Nano (rfdetr package) | Same model family as the reference; YOLO is an acceptable swap if setup is painful |
| Chord classifier | PyTorch + timm | Default backbone inception_resnet_v2; make it a config option so efficientnet_b0 can be tried for speed |
| Box labeling | Label Studio | Export fretboard boxes in COCO format |
| Config | YAML files in configs/ | Classes, paths, image size, thresholds |
| Experiment tracking | CSV logs + matplotlib plots | Keep it simple; no external services |
| Environment | uv or venv with requirements.txt | Pin versions once things work |

Starting data: the reference team's Kaggle dataset of 1,619 fretboard crops across the 5 chords, plus my own recordings (see the next sections).

## Repository structure

Use this layout; scripts are thin command-line wrappers around code in src/chordcam/.

```
chordcam/
├── SPEC.md                  # this document
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml         # classes, paths, image size, thresholds
├── data/                    # git-ignored
│   ├── raw_video/           # my recordings, one folder per chord
│   ├── frames/              # frames extracted from video
│   ├── fretboard_boxes/     # COCO labels for the detector
│   └── crops/{train,val,test}/{C,D,Em,F,G}/
├── models/                  # git-ignored weights
├── src/chordcam/
│   ├── capture.py           # webcam and video reading
│   ├── roi.py               # manual fretboard box (Milestone 2)
│   ├── detector.py          # fretboard detector wrapper
│   ├── preprocess.py        # crop, sample-wise centering, resize
│   ├── augment.py
│   ├── classifier.py        # backbone + head
│   ├── smoothing.py         # EMA and entropy rejection
│   └── overlay.py           # drawing on frames
├── scripts/
│   ├── record.py            # record labeled clips
│   ├── extract_frames.py
│   ├── make_crops.py
│   ├── train_detector.py
│   ├── train_classifier.py
│   ├── evaluate.py          # accuracy, confusion matrix, Grad-CAM
│   └── live.py              # the real-time app
└── tests/
```

## Dataset collection and labeling

The classifier trains on the Kaggle crops plus my own; the test set is mine only and is never used for training or tuning.

**Recording tool (scripts/record.py).** Shows the webcam feed and a prompt such as "Hold C — recording 20 s". It saves each clip to data/raw_video/\<chord\>/\<session\>_\<n\>.mp4, where a session name records the room and lighting (for example bedroom_lamp). It also records a "none" class: hands off the neck, changing chords, or muting strings.

**What to record.** For each chord, about 2 minutes of video across at least 4 sessions: different rooms, daylight and lamp light, and camera angles from straight on to about 30 degrees. Move the guitar a little and vary the strumming while holding the shape.

**Frames.** scripts/extract_frames.py samples 3 frames per second and drops near-duplicates (by perceptual hash) so long clips don't dominate.

**Fretboard boxes.** Label 300–500 frames in Label Studio with one box from the nut to about the 7th fret, covering the fretting hand. Mix in frames from the Kaggle set and a few YouTube screenshots for variety.

**Splits.** Split by session, never by frame, so near-identical frames never land in both train and test. Hold out at least 2 full sessions, recorded in a room not used for training, as the test set.

| Split | Sources | Target size per chord |
|---|---|---|
| Train | Kaggle train crops + most of my sessions | 300+ |
| Validation | Kaggle validation crops + 1 of my sessions | 80+ |
| Test | 2+ held-out sessions of mine, new room | 50+ |

## Model training

Two models are trained separately: the fretboard detector first, then the chord classifier on the detector's crops.

**Fretboard detector (scripts/train_detector.py).** Fine-tune RF-DETR Nano on the COCO boxes from the labeling step. Report mean average precision on a held-out set, and save a grid of predicted boxes on test frames as a PNG so I can check them by eye. Then scripts/make_crops.py runs the detector over all frames, pads each box by 5%, and writes crops into data/crops/.

**Chord classifier (scripts/train_classifier.py).** Classes are C, D, Em, F, G, and none.

- Preprocessing: sample-wise centering (subtract the image's own mean), then resize to 229×229.
- Augmentation on training data only: random rotation up to 15°, translation up to 5%, and color jitter of 0.2 on brightness, contrast, and saturation. No horizontal flips, since a flipped fretboard is a left-handed guitar.
- Model: pretrained timm backbone with its classifier removed, then global average pooling → Linear 512 → ReLU → Dropout 0.5 → Linear 128 → ReLU → Dropout 0.5 → Linear to the classes.
- Training: Adam, learning rate 0.001, cross-entropy loss. Freeze the backbone for the first 25 epochs, then unfreeze everything and drop the learning rate to 0.0001. Run up to 200 epochs with early stopping after 25 epochs without validation-loss improvement, keeping the best checkpoint.
- Save the model, the config used, and a CSV of per-epoch loss and accuracy to models/\<run_name\>/.

**Evaluation (scripts/evaluate.py).** On the test set, report overall and per-class accuracy, save a confusion matrix image, and save Grad-CAM overlays for 3 random images per class. Good attention sits on the finger and fret contact points, not the background.

## Live inference app

scripts/live.py runs the full pipeline on a webcam, a video file, or a YouTube URL (--source 0, --source clip.mp4, --source \<url\> via yt-dlp).

Per frame:

1. Read the frame and mirror it for display only, never for the model.
2. Find the fretboard: run the detector every 5th frame and reuse the last box in between, or use the manual box from roi.py with --manual-roi.
3. Crop, center, and resize exactly as in training (share the code in preprocess.py).
4. Run the classifier to get class probabilities.
5. Smooth with an exponential moving average of the probabilities (alpha 0.3, set in the config).
6. Compute normalized entropy of the smoothed probabilities. Above 0.6, or with no fretboard found, show "uncertain".
7. Draw the box colored by chord, the chord name in large text, a confidence bar, and frames per second.

Keyboard controls: q quit, r redraw the manual box, s save the current frame and crop to data/flagged/ for later labeling, and d toggle a debug panel showing the crop and all class probabilities.

Performance: print the time for each stage when --profile is set. If the app runs under 15 frames per second, try in order: run the detector less often, switch to the efficientnet_b0 backbone, and lower the capture resolution to 960×540.

## Milestones

The order gets a working live demo early, using a hand-drawn fretboard box, and adds the detector once the classifier works.

1. **Setup and webcam viewer.** Project skeleton, config loading, and live.py showing the mirrored feed with frames per second. Done when the feed runs and q quits.
2. **First classifier and manual-box demo.** Download the Kaggle crops, train the classifier on them, and add roi.py so I can drag a box around the fretboard once. Done when live.py --manual-roi names chords live, even if accuracy is rough.
3. **My own data.** record.py, extract_frames.py, the none class, session-based splits, and evaluate.py with confusion matrix and Grad-CAM. Retrain on Kaggle plus my data. Done when the test report exists and accuracy is measured on my held-out sessions.
4. **Fretboard detector.** Label boxes, train_detector.py, and make_crops.py; retrain the classifier on detector crops and switch live.py to the detector. Done when the box follows the neck as I move and accuracy holds.
5. **Stable live output.** Smoothing, entropy rejection, "uncertain" display, debug panel, flagged-frame saving, and profiling. Done when the label stops flickering and the app meets the acceptance criteria below.

## Acceptance criteria and stretch goals

The project is done when every box below is checked.

- [ ] Test accuracy of at least 85% overall and at least 75% on every chord, on held-out sessions from a room not used in training.
- [ ] live.py runs at 15 frames per second or better on my laptop with the detector on.
- [ ] The shown chord changes within 0.5 seconds of a real chord change and does not flicker while a chord is held.
- [ ] The app shows "uncertain" when no fretboard is visible or the hand is between chords.
- [ ] A new chord can be added by recording clips and rerunning documented commands, with no code changes beyond the config.
- [ ] README explains setup, recording, training, and running, and all scripts have --help.

Stretch goals, in rough order of value:

- Add audio from the mic and combine it with the video prediction, to tell apart chords that look alike, such as G and C.
- Song practice mode: load a chord progression and score whether I play the right chords in order.
- More chords: A, Am, E, Dm, and the full barre F.
- Feed short sequences of frames to a small temporal model instead of smoothing single frames.
- Port the app to the browser with ONNX Runtime Web.
