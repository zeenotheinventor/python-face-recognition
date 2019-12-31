import face_recognition
from PIL import Image, ImageDraw, ImageFont

image_of_bill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

image_of_steve = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

image_of_zeeno = face_recognition.load_image_file('./img/known/Michael Jordan.jpg')
zeeno_face_encoding = face_recognition.face_encodings(image_of_zeeno)[0]

image_of_mercy = face_recognition.load_image_file('./img/known/Mercy.jpg')
mercy_face_encoding = face_recognition.face_encodings(image_of_mercy)[0]

image_of_miriam = face_recognition.load_image_file('./img/known/Miriam.jpg')
miriam_face_encoding = face_recognition.face_encodings(image_of_miriam)[0]


image_of_elon = face_recognition.load_image_file('./img/known/Elon Musk.jpg')
elon_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_mo = face_recognition.load_image_file('./img/known/Mo.jpg')
mo_face_encoding = face_recognition.face_encodings(image_of_mo)[0]

# create array of encodings and names
known_face_encodings = [
    bill_face_encoding,
    steve_face_encoding,
    zeeno_face_encoding,
    mercy_face_encoding,
    miriam_face_encoding,
    elon_face_encoding,
    mo_face_encoding
]

known_face_names = [
    "Bill Gates",
    "Steve Jobs",
    "Zeeno",
    "Mercy",
    "Miriam",
    "Elon Mucks",
    "The Homie Mo"
]

# load test image to find faces in 
test_image = face_recognition.load_image_file('./img/groups/mary-miriam6.jpg')

# find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create an ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.45423765)

    name = "Unknown Person"

    # If match 
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    border_width = 0
    font = ImageFont.truetype("arial.ttf", 15)

    # Draw Box 
    draw.rectangle(((left-border_width, top-border_width), (right+border_width, bottom-border_width)), outline=(255,255,0), width=border_width)

    # Draw Label
    text_width, text_height = draw.textsize(name, font=font)
    draw.rectangle(((left - border_width, bottom - text_height - 10), (right + border_width, bottom)), fill=(255,255,0), outline=(255,255,0))

    draw.text((left+6, bottom - text_height - 5), name, fill=(0, 0, 0, 255), font=font)

del draw

# Display Image
pil_image.show()
pil_image.save('lul1.png')