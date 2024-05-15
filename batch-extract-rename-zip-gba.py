import os
import zipfile

# Αρχικός φάκελος με τα αρχεία zip
input_folder = r'D:\gba'

# Επεξεργασία κάθε αρχείου zip στον φάκελο
for filename in os.listdir(input_folder):
    if filename.endswith(".zip"):
        zip_file_path = os.path.join(input_folder, filename)

        # Αποσυμπίεση του αρχείου zip
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            # Εύρεση του αρχείου gba και μετονομασία σε rom.gba
            gba_file = [f for f in zip_file.namelist() if f.endswith(".gba")][0]
            zip_file.extract(gba_file, input_folder)
            os.rename(os.path.join(input_folder, gba_file), os.path.join(input_folder, "rom.gba"))

        # Διαγραφή του αρχείου zip
        os.remove(zip_file_path)

        # Συμπίεση του αρχείου rom.gba σε ένα νέο αρχείο zip με το όνομα του αρχικού zip
        new_zip_filename = os.path.splitext(filename)[0] + ".zip"
        with zipfile.ZipFile(os.path.join(input_folder, new_zip_filename), "w", zipfile.ZIP_DEFLATED) as new_zip:
            new_zip.write(os.path.join(input_folder, "rom.gba"), "rom.gba")

        # Διαγραφή του αρχείου rom.gba
        os.remove(os.path.join(input_folder, "rom.gba"))

print("Ολοκληρώθηκε η επεξεργασία των αρχείων.")