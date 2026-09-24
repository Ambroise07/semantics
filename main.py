"""
a script to run the model.
author : Gnabro Israel
date : 24 sept. 2026

"""
"""
a script to run the model.
author : Gnabro Israel
date : 24 sept. 2026
"""

import argparse
import core
from modal import TextGenerator


def main():
    # Initialize the argument parser and inject your docstring automatically
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Define command line argument flags for loading and generating
    parser.add_argument('--weights', type=str, default='virtual_usb_drive.pt', 
                        help="File path to the virtual USB drive weights file (.pt) (default: virtual_usb_drive.pt)")
    
    parser.add_argument('--count', type=int, default=50,
                        help='Number of words to reconstruct/decode from the weights file (default: 50)')

    # Parse inputs given by the user in the command prompt
    args = parser.parse_args()

    print("\n========================================================")
    print(f"🔌 Initializing Virtual USB Reader Layer...")
    print(f"📂 Source file: {args.weights}")
    print("========================================================\n")

    # 1. Instantiate a blank structural shell of the model 
    # Since we are loading weights, the rawdata argument won't be used for training.
    # We pass an empty string or dummy path to prevent reading data.md.
    model = TextGenerator(rawdata='')

    # 2. Extract weights and vocabulary from your virtual USB file (.pt)
    # This automatically updates core.C, W1, W2, stoi, itos, etc.
    model.load_from_usb(filename=args.weights)

    # 3. Decode and reconstruct text sequences purely from the math parameters space
    print(f"\n🔮 Reconstructing {args.count} words from weight matrix traces:")
    reconstructed_text = model.generate(count=args.count)
    
    print("\n--- RECONSTRUCTED TEXT OUTPUT ---")
    print(reconstructed_text)
    print("---------------------------------\n")

if __name__ == '__main__':
    main()
