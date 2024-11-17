import csv

num=25
# Input and output file paths
input_file = f'output ({num}).csv'  # Replace with the path to your input file
output_file = f'output_cleaned_{num}.csv'  # Replace with the path where you want to save the output file

# Read and clean the data
cleaned_data = []
cleaned_data.append(['ID', 'is_correct'])
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
        try:
            # Parse the row as a dictionary
            row_dict = eval(row[0])  # eval used here assuming it's safe and we trust the input format
            # Extract ID and is_correct, and clean is_correct
            cleaned_entry = [row_dict['ID'], 'True' if 'True' in row_dict['is_correct'] else 'False']
            cleaned_data.append(cleaned_entry)
        except:
            pass

# Write the cleaned data to a new file without quotes
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(cleaned_data)

print("Data cleaned and saved to", output_file)
