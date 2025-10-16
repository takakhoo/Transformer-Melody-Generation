import datetime
from pathlib import Path
import hashlib

messages = [
    "Transformers learn long-range musical dependencies via attention.",
    "Sinusoidal positional encodings preserve rhythmic structure.",
    "Autoregressive decoding enables iterative melody generation.",
    "Masking focuses loss on musical content, not padding.",
    "Multi-head attention models harmonic and rhythmic relationships.",
    "Tokenization maps pitch-duration pairs to discrete IDs.",
    "Temperature controls creativity in sampling new notes.",
    "Beam search balances exploration with coherent melodies.",
    "Data augmentation improves generalization across styles.",
    "Curriculum learning can stabilize sequence training.",
]

now = datetime.datetime.utcnow()
# Pick a message deterministically per 12h window
index = (now.timetuple().tm_yday * 2 + (0 if now.hour < 12 else 1)) % len(messages)
msg = messages[index]

# Ensure the file changes every run (timestamp + small hash)
stamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
salt = hashlib.sha1(stamp.encode()).hexdigest()[:8]

out_dir = Path("automation")
out_dir.mkdir(parents=True, exist_ok=True)

heartbeat = out_dir / "MUSIC_AI_HEARTBEAT.md"
heartbeat.write_text(
    f"# Music AI Heartbeat\n\n"
    f"- Time: {stamp}\n"
    f"- Message: {msg}\n"
    f"- Build: {salt}\n"
)

# One-liner for commit message teaser
Path(".github").mkdir(parents=True, exist_ok=True)
Path(".github/music_ai_message.txt").write_text(msg + "\n")

# Write a complete commit message that the workflow will use
commit_message = f"MusicAI: {msg} — {stamp}"
Path(".github/commit_message.txt").write_text(commit_message + "\n")


