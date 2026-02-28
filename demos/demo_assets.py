"""Generate test images with Pillow for the OmniChat demo.

No external image files needed -- everything is drawn at runtime.
"""

from PIL import Image, ImageDraw, ImageFont


def create_geometric_scene(width: int = 800, height: int = 600) -> Image.Image:
    """Create a colorful image with well-separated shapes for vision testing.

    Contains: large red circle (left), wide blue rectangle (center-right),
    green triangle (right), large yellow "HELLO WORLD" text (top),
    small orange square (bottom-right corner).
    Shapes are non-overlapping with clear proportions for accurate detection.
    """
    img = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Large dark text banner at top — centered, high contrast on light background
    try:
        font = ImageFont.truetype("arial.ttf", 56)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((width // 2 - 190, 20), "HELLO WORLD", fill=(20, 20, 20), font=font)

    # Red circle — left side, clearly round
    draw.ellipse([40, 140, 260, 360], fill=(220, 40, 40), outline=(160, 20, 20), width=3)

    # Blue rectangle — center, clearly wider than tall (4:1 ratio)
    draw.rectangle([300, 200, 620, 280], fill=(40, 70, 200), outline=(20, 40, 150), width=3)

    # Green triangle — right side, large and distinct
    draw.polygon([(660, 140), (780, 380), (540, 380)], fill=(40, 180, 70), outline=(20, 120, 40))

    # Small orange square — bottom-right corner, clearly square (equal sides)
    draw.rectangle([width - 90, height - 90, width - 30, height - 30],
                   fill=(240, 160, 40), outline=(200, 120, 20), width=2)

    # Label each shape with small text for easier verification
    try:
        label_font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        label_font = ImageFont.load_default()
    draw.text((100, 370), "Red Circle", fill=(100, 100, 100), font=label_font)
    draw.text((410, 290), "Blue Rectangle", fill=(100, 100, 100), font=label_font)
    draw.text((620, 390), "Green Triangle", fill=(100, 100, 100), font=label_font)

    return img


def create_fake_invoice(width: int = 800, height: int = 600) -> Image.Image:
    """Create a fake invoice image with a table for OCR testing.

    Contains a header, line-item table, and total. All text rendered
    with the default font (no TTF required).
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("arial.ttf", 28)
        font_med = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header
    draw.text((50, 30), "INVOICE #12345", fill=(0, 0, 0), font=font_large)
    draw.text((50, 70), "Date: 2025-01-15", fill=(100, 100, 100), font=font_small)
    draw.text((50, 90), "Customer: Acme Corporation", fill=(100, 100, 100), font=font_small)

    # Horizontal rule
    draw.line([(50, 120), (width - 50, 120)], fill=(0, 0, 0), width=2)

    # Table header
    y = 140
    cols = [50, 350, 500, 650]
    headers = ["Item", "Qty", "Price", "Total"]
    for col, header in zip(cols, headers):
        draw.text((col, y), header, fill=(0, 0, 0), font=font_med)

    # Header underline
    y += 30
    draw.line([(50, y), (width - 50, y)], fill=(180, 180, 180), width=1)

    # Table rows
    items = [
        ("Widget Pro", "10", "$25.00", "$250.00"),
        ("Gadget Plus", "5", "$49.99", "$249.95"),
        ("Connector Kit", "20", "$8.50", "$170.00"),
        ("Power Supply", "2", "$75.00", "$150.00"),
    ]
    for item_row in items:
        y += 30
        for col, cell in zip(cols, item_row):
            draw.text((col, y), cell, fill=(40, 40, 40), font=font_med)

    # Total line
    y += 50
    draw.line([(50, y), (width - 50, y)], fill=(0, 0, 0), width=2)
    y += 10
    draw.text((500, y), "TOTAL:", fill=(0, 0, 0), font=font_med)
    draw.text((650, y), "$819.95", fill=(0, 0, 0), font=font_med)

    return img
