from PIL import Image, ImageDraw, ImageFont

def stack_images_horizontally(image1: Image.Image, image2: Image.Image) -> Image.Image:
    height = max(image1.height, image2.height)
    width = image1.width + image2.width
    new_image = Image.new('RGB', (width, height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    
    return new_image

def stack_images_vertically(image1: Image.Image, image2: Image.Image, spacing: int = 0) -> Image.Image:
    width = max(image1.width, image2.width)
    height = image1.height + image2.height + spacing
    new_image = Image.new('RGB', (width, height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, image1.height + spacing))
    
    return new_image

def stack_images_horizontally_multiple(images: list, spacing: int = 10) -> Image.Image:
    """Stack multiple images horizontally with spacing between them."""
    if not images:
        raise ValueError("At least one image is required")
    
    height = max(img.height for img in images)
    width = sum(img.width for img in images) + spacing * (len(images) - 1)
    new_image = Image.new('RGB', (width, height))
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width + spacing
    
    return new_image

def add_text_label(image: Image.Image, text: str, position: str = "top", padding: int = 30, font_size: int = 24) -> Image.Image:
    """Add text label to an image with padding."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
    img_with_label = image.copy()
    draw = ImageDraw.Draw(img_with_label)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    if position == "top":
        text_x = (image.width - text_width) // 2
        text_y = padding
    elif position == "bottom":
        text_x = (image.width - text_width) // 2
        text_y = image.height - text_height - padding
    else:
        text_x = (image.width - text_width) // 2
        text_y = padding
    shadow_offset = 2
    draw.text((text_x + shadow_offset, text_y + shadow_offset), text, fill=(0, 0, 0), font=font)
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    return img_with_label