import qrcode
from PIL import Image
import os

# --- CONFIGURATION ---
LINK = "https://saferoutefrance.streamlit.app/"
LOGO_PATH = "image_0.png"  # Ensure the logo file exists in this directory
OUTPUT_NAME = "saferoute_qr_styled.png"

# Brand Colors (Extracted from Frontend CSS)
COLOR_FILL = "#0f172a"  # Dark Blue/Black (High contrast for readability)
COLOR_BG = "white"      # White background (Standard for QR scanning)

def create_styled_qr():
    """
    Generates a custom QR code with the specific brand colors
    and embeds the SafeRoute logo in the center.
    """

    # 1. QR Code Configuration
    qr = qrcode.QRCode(
        version=1,
        # IMPORTANT: 'ERROR_CORRECT_H' (High) allows up to 30% of the code
        # to be covered/damaged. This is required to place the logo
        # in the center without breaking the link.
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(LINK)
    qr.make(fit=True)

    # 2. Generate the base image with brand colors
    # .convert('RGBA') is necessary to handle the transparency of the logo later
    img_qr = qr.make_image(fill_color=COLOR_FILL, back_color=COLOR_BG).convert('RGBA')

    # 3. Embed the Logo (If file exists)
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")

            # Calculate logo size
            # We resize the logo to be 25% of the QR code's width.
            # Going larger than 30% might make the QR unreadable.
            qr_width, qr_height = img_qr.size
            logo_size = int(qr_width * 0.25)

            # High-quality resizing (LANCZOS filter)
            logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)

            # Calculate center position
            pos = ((qr_width - logo_size) // 2, (qr_height - logo_size) // 2)

            # Paste the logo
            # The third argument 'mask=logo' respects the logo's transparency
            img_qr.paste(logo, pos, mask=logo)
            print("✅ Logo successfully embedded in the center!")

        except Exception as e:
            print(f"⚠️ Error processing logo: {e}")
    else:
        print("⚠️ Logo not found. Generating a standard QR code.")

    # 4. Save to Disk
    img_qr.save(OUTPUT_NAME)
    print(f"🚀 Styled QR Code saved to: {OUTPUT_NAME}")

if __name__ == "__main__":
    create_styled_qr()
