import numpy as np

"""
    This is to be used after 'matches_combined_floor_save_CSV'.

    This adds the fiducial data to the first column in the CSV, and adds header information
"""

# Define the column headers
# header_row = ["Satellite", "Fiducial", \
#                 "F/1/1", "3/1/1", "5/1/1", "10/1/1", \
#                 "F/5/5", "3/5/5", "5/5/5", "10/5/5", \
#                 "F/10/10", "3/10/10", "5/10/10", "10/10/10", \
#                 "F/5/10", "3/5/10", "5/5/10", "10/5/10",\
#                 "F/7/10", "3/7/10", "5/7/10", "10/7/10"] 

header_row = ["Satellite", 
              "Fiducial",
              "1/1/1",
              "1/3/3",
              "1/3/5",
              "1/5/5",
              "1/7/7",
              "1/10/10",
              "3/1/1",
              "3/3/3",
              "3/3/5",
              "3/5/5",
              "3/7/7",
              "3/10/10",
              "5/1/1",
              "5/3/3",
              "5/3/5",
              "5/5/5",
              "5/7/7",
              "5/10/10",
              "7/1/1",
              "7/3/3",
              "7/3/5",
              "7/5/5",
              "7/7/7",
              "7/10/10",
              "10/1/1",
              "10/3/3",
              "10/3/5",
              "10/5/5",
              "10/7/7",
              "10/10/10"]

# Define the row headers
header_col = ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
              'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
              'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
              'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
              'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
              'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
              'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
              'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
              'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
              'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
              'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
              'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']

# Read the CSV with the floor data for the different selections
property = 'vperi_rec'
file_name = f'{property}_median'
data = np.loadtxt(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_testing_physical/{file_name}.csv", delimiter=",")

# Read in the CSV with the fiducial data
file_name_other = 'v_peri,rec'
other_data = np.loadtxt(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/paper_iii_summary/{file_name_other}-Table 1.csv", delimiter=",", skiprows=1, usecols=(1,2,3))
column_to_insert = other_data[:, 0]  # Extract the second column (index 1)

# Insert the fiducial column into the new table to be saved
data_with_insert = np.insert(data, 0, column_to_insert, axis=1)

# Add the header column to the data
data_with_col_header = np.column_stack((header_col, data_with_insert))

# Convert everything to strings for saving
data_with_headers = np.vstack((header_row, data_with_col_header))

# Save the updated array to a new CSV file
np.savetxt(
    f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{file_name}.csv",
    data_with_headers,
    delimiter=",",
    fmt="%s"
)

print("CSV file for the median data saved.")


# Now do the same thing for the widths
# Read the CSV with the floor data for the different selections
file_name = f'{property}_width'
data = np.loadtxt(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_testing_physical/{file_name}.csv", delimiter=",")

lower = other_data[:, 1]
upper = other_data[:, 2]
width = (-1)*np.ones(len(upper))
for i in range(len(width)):
    if lower[i] != -1 and upper[i] != -1:
        width[i] = upper[i]-lower[i]

# For the Nperi data
# width = other_data[:, 1]

# Insert the fiducial column into the new table to be saved
data_with_insert = np.insert(data, 0, width, axis=1)

# Add the header column to the data
data_with_col_header = np.column_stack((header_col, data_with_insert))

# Convert everything to strings for saving
data_with_headers = np.vstack((header_row, data_with_col_header))

# Save the updated array to a new CSV file
np.savetxt(
    f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{file_name}.csv",
    data_with_headers,
    delimiter=",",
    fmt="%s"
)

print("CSV file for the width data saved.")
