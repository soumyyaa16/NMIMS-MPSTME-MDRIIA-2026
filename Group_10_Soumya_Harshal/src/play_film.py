"""Play the demo film in a window.   python src\\play_film.py [path.mp4]
Keys: SPACE pause/play | R restart | F fullscreen | LEFT/RIGHT (A/D) skip 5 s | Q or ESC quit"""
import sys
import time
from pathlib import Path

import cv2

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "film" / "demo_film.mp4"
if not path.exists():
    raise SystemExit(f"File not found: {path}\nRender it first:  python src\\make_film.py --case c160 --scenario enhancing_superior --nifti D:\\robotics\\brats_cases")
cap = cv2.VideoCapture(str(path))
if not cap.isOpened():
    raise SystemExit(f"Could not open {path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{path.name}: {total} frames, {fps:.1f} fps, {total / fps:.0f} s   (SPACE pause, R restart, F fullscreen, Q quit)")
WIN = "Demo film"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)
full = False
paused = False
frame = None
nxt = time.time()
while True:
    if not paused or frame is None:
        ok, frame = cap.read()
        if not ok:                       # end of film: hold the last frame until a key is pressed
            paused = True
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
            ok, frame = cap.read()
            if frame is None:
                break
    cv2.imshow(WIN, frame)
    nxt += 1.0 / fps
    wait = max(1, int((nxt - time.time()) * 1000))
    if wait > 1000 or nxt < time.time() - 1:
        nxt = time.time()
    key = cv2.waitKeyEx(wait)
    if key in (ord("q"), 27):
        break
    if key == ord(" "):
        paused = not paused
        nxt = time.time()
    elif key in (ord("r"), ord("R")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        paused = False
        nxt = time.time()
    elif key in (ord("f"), ord("F")):
        full = not full
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    elif key in (2424832, ord("a"), ord("A"), 2555904, ord("d"), ord("D")):      # left / right arrows
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        step = int(5 * fps) * (-1 if key in (2424832, ord("a"), ord("A")) else 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, pos + step)))
        nxt = time.time()
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()