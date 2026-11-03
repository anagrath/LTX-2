# JE A/B edit with Drive B-roll

Owner selected Take B (seed43) as base, B-roll on the unexpected Really, return to
walking JE on have in We have seen success, and Take A (seed42) picture/audio for
Do me a favor, go check it out okay? They later clarified Really must be removed,
not only hidden by B-roll. Initial edit failed that requirement and owner reported
it appeared fast-forwarded. The initial file is preserved only on scratch; the
corrected export replaced it in Shared/Videos. A/B/Fu Lu originals are preserved.

B-roll was discovered via Google Drive skill/plugin in Elephant Learning B-Rolls,
verified in marketing-asset-index.csv (metadata_classified b_roll), file
1NvqtYBFnQ0inut33ER1EFhImCOHuiKsQ, Students Using Elephant Learning App.mp4.
Connector raw download returned413; configured rclone fetched the1441211989-byte
file read-only using its verified folder1FwmxpBGInbSFo47nipAlBxeYK11kSMmY. It is
1920x1080,24000/1001fps,570.0695s. The full asset index, raw B-roll and contact
sheets remain outside Git; no Drive writes/sharing/publishing occurred.

Corrected timeline: B frames0:267, B-roll over B audio from source318:517, return
to B picture/audio517:935, then A918:985. This removes B267:318 (11.125–13.25s),
including Really, while retaining a short lead-in before We accept. B-roll source
30s for96frames crops608x1080 at950,0; source300s for103frames crops608x1080 at850,0.
Both scale to576x1024 and24fps; B-roll audio is omitted. Return is output frame466
(19.4167s), source B21.5417s at have. A closing begins output frame884 (36.8333s),
source A38.25s. Word times are Whisper estimates; raw cue words and source preview
frames were inspected. Source input files remain unchanged.

No time scaling or tempo filter: each picture/audio segment keeps source timing,
with only timestamp rebasing, cuts, crop/scale, and5ms audio declick fades at the
Really removal and A closing cut. Corrected951frames/24fps =39.625s, H264/AAC,
576x1024,48k stereo. Corrected encode7.02s on CPU. QuickTime was closed for the
old edited document, reopened for corrected file, reset to0s and explicit1x;
AppleScript returned1.0 and39.625. User perceptual review is still required.

For context, initial rejected edit was1002frames/24fps=41.75s; same-time decoded
voice correlation against B was0.9999777 at1–10s and0.9999868 at22–37s. That does
not explain or negate the owner's perceived speed issue. Corrected edit rebuilt
segments from original A/B rather than stacking another edit on the rejected file.

Current review output: Shared/Videos/JE All Ages - Edited A+B.mp4. No upload or
asset-index write was requested. Files here preserve scripts and timing evidence,
not a claim of final creative approval or word-perfect transcription.

## Owner-directed tighter revision

Owner identified a long gap near17s, requested collapsing it and returning on
seen (rather than have), then directed switching to A at If your student is
working on and using A through the end. Silence detection in the39.625s previous
cut found16.749396–19.203104s below-32dB, matching that reported gap.

Latest edit rebuilt from original sources: B0:267; B-roll over B audio318:456
then508:522; B picture/audio522:751; A picture/audio745:985. Removes the Really
interval267:318 plus the gap456:508 (B19–21.1667s), preserving about0.29s natural
pause. B-roll ends at output419/24=17.4583s, sourceB21.75s on seen. A begins at
output648/24=27s, sourceA31.0417s at the start of its If your student window;
source B cuts31.2917s in its short pause before If. A continues through the closing.

The two B-roll crops now use96 and56frames. No tempo/time-scaling filters.
888frames/24fps=37s; encode6.158s. Open/play commands returned1x and37s on Mac.
Latest review output: Shared/Videos/JE All Ages - Final Cut.mp4. Previous corrected
edit moved back to scratch, so Shared does not retain that superseded review file.
OriginalA/B/Fu Lu remain. No new ASR or full-decode review delayed playback, and
owner approval remains pending. New scripts/timeline are edit_tightened.py and
tightened-timeline.json; earlier scripts are historical evidence.
