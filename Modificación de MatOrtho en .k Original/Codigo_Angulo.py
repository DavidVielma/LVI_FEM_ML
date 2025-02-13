import shutil

def copy_file(input_file):
    # Define the path for the copied file
    copied_file = input_file.replace('.k', '_copy.k')
    
    # Copy the original file to the new file
    shutil.copy(input_file, copied_file)
    
    return copied_file

def update_coordinates(input_file, output_file, num_blocks=120000, num_blocks_per_group=15000):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    
    def update_block(block_number, cosply, sinply, sinplyneg):
        # First line of the block (ID line) - no tabulation
        updated_lines.append(lines[block_number * 3])
        
        # Second line of the block (coordinates x, y, z) with tabulation
        updated_lines.append(f"{cosply}       {sinply}       0.0\n")
        
        # Third line of the block (coordinates a, b, c) with tabulation
        updated_lines.append(f"{sinplyneg}       {cosply}       0.0\n")

    for i in range(0, num_blocks, num_blocks_per_group):
        # Determine the current block group
        group_index = (i // num_blocks_per_group) + 1
        
        # Define the variables for the current block group
        cosply = f"cosply{group_index}"
        sinply = f"sinply{group_index}"
        sinplyneg = f"sinplyneg{group_index}"  # Nueva variable explícita para el valor negativo
        
        # Update blocks in the current group
        for block_number in range(i, i + num_blocks_per_group):
            update_block(block_number, cosply, sinply, sinplyneg)

    with open(output_file, 'w') as file:
        file.writelines(updated_lines)
    
    print(f"Archivo modificado guardado como: {output_file}")

# Define file paths
input_file = r'C:\Users\peped\OneDrive - usach.cl\Univeridad\13._Tesis\Simulaciones\0.-Casos Tesis\Caso Parametrizado\Pre archivos\MatOrtho.k'  # Replace with your input file path

# Create a copy of the original file
copied_file = copy_file(input_file)

# Update the copied file
update_coordinates(input_file, copied_file)
