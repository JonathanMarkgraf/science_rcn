import requests
from io import BytesIO
from PIL import Image

from science_rcn import run



r = requests.get('CAPTCHA_URL')
b = BytesIO(r.content)
i = Image.open(b)
i.save('./test.png')
print(run.readCaptcha(i))