from PIL import Image, ImageDraw

def create_blank_mannequin():
    # Create the standard 3:4 aspect ratio image for IDM-VTON (e.g., 768x1024)
    img = Image.new('RGB', (768, 1024), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # Draw a very faint, abstract "invisible mannequin" shape 
    # so IDM-VTON knows roughly where to place the garment
    # Neck / shoulders
    draw.ellipse((284, 150, 484, 300), fill=(240, 240, 240))
    # Torso
    draw.rectangle((234, 250, 534, 850), fill=(240, 240, 240))
    # Hips/Legs
    draw.rectangle((234, 800, 534, 1024), fill=(240, 240, 240))
    
    img.save("blank_mannequin.jpg", quality=90)
    print("Created blank_mannequin.jpg")

if __name__ == "__main__":
    create_blank_mannequin()
