import os

# Pfad zum Ordner mit den Bildern
ordner_pfad = './dataset/resized/resized'

# Durchlaufe alle Dateien im Ordner
for datei_name in os.listdir(ordner_pfad):
    # Überprüfe, ob der Dateiname den zu ersetzenden String enthält
    if 'Albrecht_Du╠êrer' in datei_name:
        # Erstelle den neuen Dateinamen
        neuer_datei_name = datei_name.replace('Albrecht_Du╠êrer', 'Albrecht_Duerer')
        # Vollständige Pfade der alten und neuen Datei
        alter_pfad = os.path.join(ordner_pfad, datei_name)
        neuer_pfad = os.path.join(ordner_pfad, neuer_datei_name)
        # Benenne die Datei um
        os.rename(alter_pfad, neuer_pfad)
        print(f"'{datei_name}' wurde umbenannt zu '{neuer_datei_name}'")

print("Alle Dateien wurden umbenannt.")