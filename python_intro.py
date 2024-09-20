import face_recognition
import numpy as np
from PIL import Image, ImageFilter, ImageDraw



def main():
    
    find_individual_face_in_group("derek.jpg", "dkitantwerp.jpg")

def helloworld():
    print("Hello, World!")

def blur(image):
    before = Image.open(image)
    after = before.filter(ImageFilter.BoxBlur(10))
    after.save("blurred.jpg")

def find_faces(image):
    image_to_search = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(image_to_search)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image_to_search[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()

def find_individual_face_in_group(individual_face_image, group_image):
    known_face = face_recognition.load_image_file(individual_face_image)
    known_face_encoding = face_recognition.face_encodings(known_face)[0]
    image_to_search = face_recognition.load_image_file(group_image)
    face_locations = face_recognition.face_locations(image_to_search)
    face_encodings = face_recognition.face_encodings(image_to_search, face_locations)
    pil_image = Image.open(group_image)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), face_encodings in zip(face_locations,
                                                          face_encodings):
        matches = face_recognition.compare_faces([known_face_encoding], face_encodings)
        face_distances = face_recognition.face_distance([known_face_encoding], face_encodings)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            draw.rectangle(((left, top), (right, bottom)), outline = (255,0,0), width = 5)
    del draw
    pil_image.show()

  

if __name__ == "__main__":
    main()