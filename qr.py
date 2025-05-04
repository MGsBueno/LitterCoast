import qrcode

# Link que você deseja transformar em QR Code
link = "https://forms.gle/PLDgtbQkSxKboPfv7"

# Cria o objeto QRCode
qr = qrcode.QRCode(
    version=1,  # controla o tamanho do QR Code (1 a 40)
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # nível de correção de erro
    box_size=10,  # tamanho de cada caixa do QR Code
    border=4,  # borda ao redor do QR Code
)

# Adiciona o link ao QRCode
qr.add_data(link)
qr.make(fit=True)

# Cria a imagem do QRCode
img = qr.make_image(fill_color="black", back_color="white")

# Salva como imagem
img.save("qrcode_link.png")

print("QR Code gerado e salvo como 'qrcode_link.png'")
