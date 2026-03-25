import requests
import pandas as pd
from pymatgen.ext.matproj import MPRester as LegacyMPRester
from mp_api.client import MPRester
from ase.io import read as ase_read
from ase.db import connect as ase_connect
from pymatgen.io.ase import AseAtomsAdaptor
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()


def filter_lowest_energy_structures(data):
    """
    Filter data to keep only the structure with lowest energy_above_hull for each formula.
    
    Args:
        data: List of material documents from Materials Project
        
    Returns:
        Filtered list with only lowest energy structures for each formula
    """
    if not data:
        return data
    
    formula_to_best = {}
    
    for item in data:
        if not hasattr(item, 'formula_pretty') or not hasattr(item, 'energy_above_hull'):
            continue
            
        formula = item.formula_pretty
        energy = item.energy_above_hull
        
        # Skip if energy_above_hull is None
        if energy is None:
            continue
            
        # Keep the structure with lowest energy_above_hull for this formula
        if formula not in formula_to_best or energy < formula_to_best[formula]['energy']:
            formula_to_best[formula] = {
                'energy': energy,
                'item': item
            }
    
    # Extract the best items
    filtered_data = [entry['item'] for entry in formula_to_best.values()]
    
    logging.info(f"Filtered from {len(data)} to {len(filtered_data)} materials (lowest energy per formula)")
    return filtered_data


def fetch_data_from_mp(keep_lowest_energy_only=True):
    """
    Fetch data from Materials Project using API key from .env file.
    
    Args:
        keep_lowest_energy_only: If True, only keep the structure with lowest energy_above_hull for each formula
    """
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        logging.error("MP_API_KEY not found in .env file.")
        return []
    try:
        with MPRester(api_key) as mpr:
            if keep_lowest_energy_only:
                # API层面预筛选，只获取稳定的材料
                logging.info("Using API pre-filtering for stable materials only...")
                data = mpr.materials.search(
                    energy_above_hull=(None, 0.1),  # 只获取相对稳定的材料
                    nelements=(1, 4),  # 限制元素数量
                    fields=["material_id", "formula_pretty", "structure", "dimensionality", "energy_above_hull", "formation_energy_per_atom"]
                )
                logging.info(f"API pre-filtering fetched {len(data)} stable materials.")
                
                # 进一步筛选每个化学式的最低能量结构
                data = filter_lowest_energy_structures(data)
                logging.info(f"After per-formula filtering: {len(data)} unique lowest energy structures.")
            else:
                # 获取所有材料（原始行为）
                data = mpr.materials.search(
                    fields=["material_id", "formula_pretty", "structure", "dimensionality", "energy_above_hull", "formation_energy_per_atom"]
                )
                logging.info(f"Successfully fetched {len(data)} entries from Materials Project.")
            
            return data
    except Exception as e:
        logging.error(f"Error fetching data from Materials Project: {e}")
        return []


def fetch_data_from_c2db(db_path="data/raw/c2db.db"):
    """
    Fetch data from a local C2DB ASE database file.
    The user is expected to download 'c2db.db' and place it in db_path.
    Extracts structures and relevant properties like dgH and hform.
    """
    if not os.path.exists(db_path):
        logging.warning(f"C2DB database file not found at {db_path}. "
                        f"Please download it from https://cmr.fysik.dtu.dk/c2db/c2db.html and place it in {db_path}.")
        return []
    
    materials_data = []
    try:
        db = ase_connect(db_path)
        for row in db.select(): # Select all available rows and their properties
            try:
                atoms = row.toatoms() # Get ASE Atoms object
                # Extract relevant properties, using row.get(key, default_value) for safety
                material_entry = {
                    "atoms": atoms,
                    "c2db_id": row.id,
                    "formula": row.formula,
                    "dgH": row.get('dgH'),  # Key for Gibbs free energy of H adsorption
                    "hform": row.get('hform'), # Key for formation energy
                    "band_gap_pbe": row.get('gap'), # PBE band gap from example
                    "band_gap_hse": row.get('gap_hse'), # HSE band gap from example
                    "band_gap_gw": row.get('gap_gw'),   # GW band gap from example
                    # Add other potentially useful C2DB specific key-value pairs
                    # You might want to inspect `row.key_value_pairs` to see all available data
                }
                # Example to add all other key_value_pairs, be mindful of data size
                # for key, value in row.key_value_pairs.items():
                #     if key not in material_entry: # avoid overwriting defined keys
                #         material_entry[f"c2db_{key}"] = value

                materials_data.append(material_entry)
            except Exception as e_row:
                logging.warning(f"Error processing a row from C2DB (id: {row.id if hasattr(row, 'id') else 'unknown'}): {e_row}")
                continue # Skip to the next row if one row fails
        logging.info(f"Successfully loaded and parsed {len(materials_data)} entries from C2DB at {db_path}.")
    except Exception as e:
        logging.error(f"Error reading C2DB database at {db_path}: {e}")
        return []
    return materials_data

def fetch_data_from_mc2d():
    """Fetch data from MC2D database. Placeholder - requires investigation on data access."""
    logging.info("Fetching data from MC2D (Placeholder - to be implemented).")
    return []

def fetch_data_from_2dmatpedia():
    """Fetch data from 2DMatPedia database. Placeholder - requires investigation on data access."""
    logging.info("Fetching data from 2DMatPedia (Placeholder - to be implemented).")
    return []


def standardize_structure(struct_obj):
    """
    Standardizes a structure object.
    Converts ASE Atoms to Pymatgen Structure if necessary.
    Can be expanded with more standardization routines.
    """
    if hasattr(struct_obj, 'get_potential_energy'): # Heuristic for ASE Atoms object
        try:
            return AseAtomsAdaptor.get_structure(struct_obj)
        except Exception as e:
            logging.error(f"Error converting ASE Atoms to Pymatgen Structure: {e}")
            return None
    elif hasattr(struct_obj, 'lattice') and hasattr(struct_obj, 'sites'): # Heuristic for Pymatgen Structure
        return struct_obj
    else:
        logging.warning(f"Unrecognized structure object type: {type(struct_obj)}")
        return None

def preprocess_data(raw_data_list, source_name="Unknown"):
    """
    Preprocesses a list of raw structure data.
    - Standardizes structures (e.g., converts ASE to Pymatgen).
    - Extracts basic information and specific properties.
    Returns a list of dictionaries with processed data.
    """
    processed_list = []
    if not raw_data_list:
        logging.info(f"No data to preprocess for {source_name}.")
        return pd.DataFrame()

    for i, item in enumerate(raw_data_list):
        struct_to_standardize = None
        material_id_val = None
        formula_val = None
        extra_props = {}

        if source_name == "MP":
            if hasattr(item, 'structure'):
                struct_to_standardize = item.structure
                material_id_val = str(item.material_id) if hasattr(item, 'material_id') else f"mp_gen_{i}"
                formula_val = item.formula_pretty if hasattr(item, 'formula_pretty') else None
                extra_props['dimensionality'] = getattr(item, 'dimensionality', None)
                extra_props['energy_above_hull'] = getattr(item, 'energy_above_hull', None)
            else:
                logging.warning(f"MP item (index {i}) does not have 'structure' attribute: {item}")
                continue
        elif source_name == "C2DB":
            # item is a dictionary from fetch_data_from_c2db
            struct_to_standardize = item.get('atoms')
            material_id_val = str(item.get('c2db_id', f"c2db_gen_{i}"))
            formula_val = item.get('formula')
            extra_props['dgH'] = item.get('dgH')
            extra_props['hform'] = item.get('hform')
            extra_props['band_gap_pbe'] = item.get('band_gap_pbe')
            extra_props['band_gap_hse'] = item.get('band_gap_hse')
            extra_props['band_gap_gw'] = item.get('band_gap_gw')
            # You can add more C2DB specific props here
        else: # For MC2D, 2DMatPedia or other future sources
            # This part needs to be adapted based on how data from these sources is fetched
            struct_to_standardize = item # Assuming item is already a structure object or can be processed
            material_id_val = f"{source_name.lower()}_gen_{i}"
            # formula_val would need to be derived or fetched

        if struct_to_standardize is None:
            logging.warning(f"No structure found for item {material_id_val if material_id_val else i} from {source_name}.")
            continue
            
        standardized_struct = standardize_structure(struct_to_standardize)
        
        if standardized_struct:
            if formula_val is None: # Try to get formula from standardized structure if not already set
                formula_val = standardized_struct.composition.reduced_formula
            
            entry = {
                "material_id": material_id_val,
                "formula": formula_val,
                "structure_obj": standardized_struct, # Store the Pymatgen object for further use
                "source": source_name,
            }
            entry.update(extra_props) # Add all other extracted properties
            processed_list.append(entry)
        else:
            logging.warning(f"Could not standardize structure for item {material_id_val if material_id_val else i} from {source_name}.")

    logging.info(f"Successfully preprocessed {len(processed_list)} entries from {source_name}.")
    return pd.DataFrame(processed_list)


def extract_2d_subset(all_processed_data_df):
    """
    Extracts a subset of 2D materials from the processed data.
    This is a placeholder and needs specific logic based on available data
    (e.g., 'dimensionality' field from MP, or relying on the source being a 2D DB).
    """
    logging.info("Attempting to extract 2D materials subset...")
    
    if all_processed_data_df.empty:
        logging.info("Input DataFrame for 2D subset extraction is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    is_2d_flags = []
    # Ensure required columns exist before iterating
    if not ('structure_obj' in all_processed_data_df.columns and \
            'source' in all_processed_data_df.columns):
        logging.warning("'structure_obj' or 'source' column missing. Cannot reliably determine 2D materials. Returning original DataFrame.")
        return all_processed_data_df.copy()

    for index, row in all_processed_data_df.iterrows():
        is_2d = False # Default to False
        if row['source'] == "MP":
            if 'dimensionality' in row and row['dimensionality'] == 2.0:
                is_2d = True
        elif row['source'] in ["C2DB", "MC2D", "2DMatPedia"]: 
            is_2d = True
        is_2d_flags.append(is_2d)
    
    if len(is_2d_flags) == len(all_processed_data_df):
        filtered_df = all_processed_data_df[is_2d_flags].copy()
        logging.info(f"Extracted {len(filtered_df)} 2D materials from a total of {len(all_processed_data_df)} entries.")
        return filtered_df
    else: 
        logging.warning("Mismatch in 2D flag generation. Could not reliably determine 2D subset. Returning original DataFrame.")
        return all_processed_data_df.copy()

def main():
    # Ensure processed data directory exists
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True) # Ensure raw data directory exists for C2DB

    # --- Data Fetching ---
    data_mp_raw = fetch_data_from_mp(keep_lowest_energy_only=True)  # 只保留最低能量结构
    data_c2db_raw = fetch_data_from_c2db() # Expects data/raw/c2db.db
    data_mc2d_raw = fetch_data_from_mc2d() # Placeholder
    data_2dmatpedia_raw = fetch_data_from_2dmatpedia() # Placeholder
    
    # --- Data Preprocessing ---
    df_mp = preprocess_data(data_mp_raw, source_name="MP")
    df_c2db = preprocess_data(data_c2db_raw, source_name="C2DB")
    df_mc2d = preprocess_data(data_mc2d_raw, source_name="MC2D") # Will be empty if placeholder not implemented
    df_2dmatpedia = preprocess_data(data_2dmatpedia_raw, source_name="2DMatPedia") # Will be empty

    # --- Combine and Save ---
    # Save individual processed dataframes
    if not df_mp.empty:
        df_mp.to_pickle('data/processed/mp_processed.pkl') # Using pickle to preserve Structure objects
        logging.info("Saved Materials Project processed data to data/processed/mp_processed.pkl")
    if not df_c2db.empty:
        df_c2db.to_pickle('data/processed/c2db_processed.pkl')
        logging.info("Saved C2DB processed data to data/processed/c2db_processed.pkl")
    if not df_mc2d.empty:
        df_mc2d.to_pickle('data/processed/mc2d_processed.pkl')
        logging.info("Saved MC2D processed data to data/processed/mc2d_processed.pkl")
    if not df_2dmatpedia.empty:
        df_2dmatpedia.to_pickle('data/processed/2dmatpedia_processed.pkl')
        logging.info("Saved 2DMatPedia processed data to data/processed/2dmatpedia_processed.pkl")

    # Combine all dataframes
    all_data_frames = [df for df in [df_mp, df_c2db, df_mc2d, df_2dmatpedia] if not df.empty]
    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        combined_df.to_pickle('data/processed/all_materials_processed.pkl')
        logging.info(f"Saved combined processed data ({len(combined_df)} entries) to data/processed/all_materials_processed.pkl")

        # --- Extract 2D materials subset for fine-tuning ---
        # This step might need more sophisticated filtering based on structure analysis
        # if 'dimensionality' is not consistently available or reliable from MP.
        fine_tuning_set_df = extract_2d_subset(combined_df.copy()) # Pass a copy
        fine_tuning_set_df.to_pickle('data/processed/fine_tuning_set_2d.pkl')
        logging.info(f"Saved 2D materials subset for fine-tuning ({len(fine_tuning_set_df)} entries) to data/processed/fine_tuning_set_2d.pkl")
    else:
        logging.info("No data processed from any source. Skipping combination and subset extraction.")
    
    logging.info("Data acquisition and preprocessing step finished.")


if __name__ == "__main__":
    main() 