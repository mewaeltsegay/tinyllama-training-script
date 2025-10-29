#!/usr/bin/env python3
"""
Script to check the quality of the original dataset files.
"""

import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_original_dataset():
    """Check the quality of original dataset files."""
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    for split in ["train", "validation", "test"]:
        file_path = dataset_dir / f"{split}.jsonl"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"\n=== Checking {split}.jsonl ===")
        
        # Read first 10 lines
        line_count = 0
        valid_lines = 0
        repetitive_lines = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    if i >= 10:  # Only check first 10 lines for detailed analysis
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        valid_lines += 1
                        
                        logger.info(f"\nLine {i+1}:")
                        logger.info(f"  Length: {len(text)} characters")
                        logger.info(f"  Text preview: '{text[:100]}...'")
                        
                        # Check for repetition
                        if len(text) > 10:
                            # Count most common character
                            char_counts = {}
                            for char in text:
                                char_counts[char] = char_counts.get(char, 0) + 1
                            
                            if char_counts:
                                most_common_char = max(char_counts, key=char_counts.get)
                                most_common_count = char_counts[most_common_char]
                                repetition_ratio = most_common_count / len(text)
                                
                                logger.info(f"  Most common char: '{most_common_char}' ({most_common_count}/{len(text)} = {repetition_ratio:.1%})")
                                
                                if repetition_ratio > 0.7:
                                    logger.warning(f"  ⚠️  HIGH REPETITION: {repetition_ratio:.1%} of text is '{most_common_char}'")
                                    repetitive_lines += 1
                                
                                # Check for actual Tigrinya content
                                tigrinya_chars = set('ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆቈቊቋቌቍበቡቢባቤብቦቨቩቪቫቬቭቮተቱቲታቴትቶቸቹቺቻቼችቾኀኁኂኃኄኅኆኈኊኋኌኍነኑኒናኔንኖኘኙኚኛኜኝኞአኡኢኣኤእኦከኩኪካኬክኮኰኲኳኴኵኸኹኺኻኼኽኾዀዂዃዄዅወዉዊዋዌውዎዐዑዒዓዔዕዖዘዙዚዛዜዝዞዠዡዢዣዤዥዦየዩዪያዬይዮደዱዲዳዴድዶዸዹዺዻዼዽዾጀጁጂጃጄጅጆገጉጊጋጌግጎጐጒጓጔጕጠጡጢጣጤጥጦጨጩጪጫጬጭጮጰጱጲጳጴጵጶጸጹጺጻጼጽጾፀፁፂፃፄፅፆፈፉፊፋፌፍፎፐፑፒፓፔፕፖ')
                                tigrinya_count = sum(1 for char in text if char in tigrinya_chars)
                                tigrinya_ratio = tigrinya_count / len(text) if len(text) > 0 else 0
                                
                                logger.info(f"  Tigrinya chars: {tigrinya_count}/{len(text)} = {tigrinya_ratio:.1%}")
                                
                                if tigrinya_ratio < 0.3:
                                    logger.warning(f"  ⚠️  LOW TIGRINYA CONTENT: Only {tigrinya_ratio:.1%} Tigrinya characters")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"  ❌ Invalid JSON on line {i+1}: {e}")
                    except Exception as e:
                        logger.error(f"  ❌ Error processing line {i+1}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            continue
        
        # Count total lines in file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            logger.info(f"\n=== Summary for {split}.jsonl ===")
            logger.info(f"Total lines: {total_lines}")
            logger.info(f"Valid lines (first 10): {valid_lines}/10")
            logger.info(f"Repetitive lines (first 10): {repetitive_lines}/10")
            
            if repetitive_lines > 5:
                logger.error(f"❌ DATASET ISSUE: {repetitive_lines}/10 lines are highly repetitive")
                logger.error("This dataset appears to be corrupted or contains test data")
            elif repetitive_lines > 0:
                logger.warning(f"⚠️  Some repetitive content found: {repetitive_lines}/10 lines")
            else:
                logger.info(f"✅ Dataset quality looks good")
                
        except Exception as e:
            logger.error(f"Error counting lines in {file_path}: {e}")
    
    return True

if __name__ == "__main__":
    success = check_original_dataset()
    if success:
        print("\n✅ Dataset quality check completed. See logs above for issues.")
    else:
        print("\n❌ Dataset quality check failed!")