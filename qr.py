import qrcode

# Link to convert into a QR code.
link = "https://forms.gle/PLDgtbQkSxKboPfv7"

# Create the QRCode object.
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# Add the link to the QR code.
qr.add_data(link)
qr.make(fit=True)

# Create the QR code image.
image = qr.make_image(fill_color="black", back_color="white")

# Save the generated image.
image.save("qrcode_link.png")

print("QR code generated and saved as 'qrcode_link.png'")
