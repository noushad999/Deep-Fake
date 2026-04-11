"""Check what timm models are available locally."""
import timm

print("StyleGAN models:")
sg = timm.list_models('*stylegan*')
print(f"  {sg}")

print("\nGAN models:")
gan = timm.list_models('*gan*')
print(f"  {gan[:30]}")

print("\nFace models:")
face = timm.list_models('*face*')
print(f"  {face}")

print("\nSynthesis models:")
syn = timm.list_models('*synthes*')
print(f"  {syn}")

# Try loading any GAN model
if gan:
    print(f"\nTrying to load: {gan[0]}")
    try:
        m = timm.create_model(gan[0], pretrained=True)
        print(f"  Success! {m}")
    except Exception as e:
        print(f"  Failed: {e}")
