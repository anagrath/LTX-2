from pathlib import Path
import json, shutil, subprocess, time

root = Path(__file__).parent
while not (root/'driver-timing.json').exists():
    if next((root/'queue').glob('*.failed.json'),None):
        raise RuntimeError('Render failed; inspect queue receipt')
    time.sleep(.5)
source = Path('/home/anagrath/ai/outputs/ltx/je-refined-standing-41-20261102/video.mp4')
destination = Path('/mnt/mongo-raid/shared/Videos/JE All Ages - Refined Standing - 41 Seconds.mp4')
if destination.exists():
    raise FileExistsError(destination)
shutil.move(source,destination)
source.symlink_to(destination)
subprocess.run(['ssh','mac','open -a "QuickTime Player" "/Volumes/Shared/Videos/JE All Ages - Refined Standing - 41 Seconds.mp4"'],check=True)
apple = '''osascript -e 'tell application "QuickTime Player"' -e 'activate' -e 'set movieDocument to open POSIX file "/Volumes/Shared/Videos/JE All Ages - Refined Standing - 41 Seconds.mp4"' -e 'play movieDocument' -e 'end tell' '''
subprocess.run(['ssh','mac',apple],check=True)
receipt = {'shared_video':str(destination),'mac_open_and_play_commands_succeeded':True,'bytes':destination.stat().st_size}
(root/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
