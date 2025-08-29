
# Simple bash script to create a bunch of directories

for dirname in "Aquarius II" "Aquarius III" "Bootes I" "Bootes II" "Bootes III" "Bootes IV" "Bootes V" "Canes Venatici I" "Canes Venatici II" "Carina" "Carina II" "Carina III" "Centaurus I" "Cetus II" "Cetus III" "Columba I" "Coma Berenices" "Crater II" "Draco" "Draco II" "Eridanus II" "Eridanus III" "Eridanus IV" "Fornax" "Grus I" "Grus II" "Hercules" "Horologium I" "Horologium II" "Hydra II" "Hydrus I" "Indus I" "Leo I" "Leo II" "Leo IV" "Leo V" "Leo VI" "Leo A" "Leo T" "Leo Minor I" "Pegasus III" "Pegasus IV" "Phoenix I" "Phoenix II" "Pictor I" "Pictor II" "Pisces II" "Reticulum II" "Reticulum III" "Sagittarius" "Sagittarius II" "Sculptor" "Segue 1" "Segue 2" "Sextans" "Sextans II" "Triangulum II" "Tucana I" "Tucana II" "Tucana III" "Tucana IV" "Tucana V" "Ursa Major I" "Ursa Major II" "Ursa Minor" "Virgo I" "Virgo II" "Virgo III" "Willman 1"; do
    sanitized_dirname=${dirname// /_}
    mkdir "$sanitized_dirname"
done

floor_list = [(1, 1, 1),
              (3, 1, 1),
              (5, 1, 1),
              (7, 1, 1),
              (10, 1, 1),
              (1, 3, 3),
              (3, 3, 3),
              (5, 3, 3),
              (7, 3, 3),
              (10, 3, 3),
              (1, 5, 5),
              (3, 5, 5),
              (5, 5, 5),
              (7, 5, 5),
              (10, 5, 5),
              (1, 7, 7),
              (3, 7, 7),
              (5, 7, 7),
              (7, 7, 7),
              (10, 7, 7),
              (1, 10, 10),
              (3, 10, 10),
              (5, 10, 10),
              (7, 10, 10),
              (10, 10, 10),
              (1, 3, 5),
              (3, 3, 5),
              (5, 3, 5),
              (7, 3, 5),
              (10, 3, 5)]

