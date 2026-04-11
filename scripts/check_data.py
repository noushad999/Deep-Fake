from pathlib import Path

real = list(Path("/mnt/e/deepfake-detection/data_faces/real").glob("*.jpg"))
fake = list(Path("/mnt/e/deepfake-detection/data_faces/fake").glob("*.jpg"))
print(f"Real: {len(real)}")
print(f"Fake: {len(fake)}")
print(f"Total: {len(real) + len(fake)}")
print(f"\nSample real: {real[0].name}, size={real[0].stat().st_size/1024:.0f}KB")
print(f"Sample fake: {fake[0].name}, size={fake[0].stat().st_size/1024:.0f}KB")
