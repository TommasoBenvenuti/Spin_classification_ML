def target_single_ion(target_spin_values, target_coord_no, target_metal_z ):
# così sembra non essere troppo utile questa funzione
# mi serve per vedere se la struttura analizzata ha o meno un metallo di interesse
    collected = {s: 0 for s in target_spin_values} #set comprehension
    print("Target spin values:", target_spin_values)
    print("Target coordination number:", target_coord_no)
    print("Target metal atomic number (Z):", target_metal_z)
    
    return target_spin_values, target_coord_no, collected, target_metal_z
