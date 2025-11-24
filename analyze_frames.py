#!/usr/bin/env python3
"""
Analyze extracted video frames using OpenAI's Vision API.
"""

import os
import base64
import argparse
from pathlib import Path
from openai import OpenAI


def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_images_from_directory(directory, max_images=None):
    """
    Load image files from a directory.
    
    Args:
        directory (str): Path to directory containing images
        max_images (int): Maximum number of images to load (None for all)
    
    Returns:
        list: List of image file paths sorted by name
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    image_paths = []
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    # Get all image files
    for file_path in sorted(directory_path.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {directory}")
    
    # Limit number of images if specified
    if max_images:
        image_paths = image_paths[:max_images]
    
    return image_paths


def analyze_frames_with_openai(image_paths, prompt, model="gpt-4o", api_key=None, detail="auto"):
    """
    Analyze images using OpenAI's Vision API.
    
    Args:
        image_paths (list): List of paths to image files
        prompt (str): Text prompt for the analysis
        model (str): OpenAI model to use (default: gpt-4o)
        api_key (str): OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        detail (str): Image detail level: "low", "high", or "auto" (default: "auto")
    
    Returns:
        str: API response text
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    print(f"\n🔧 Preparing to analyze {len(image_paths)} images...")
    print(f"   Model: {model}")
    print(f"   Detail level: {detail}\n")
    
    # Prepare the messages for the API
    content = [{"type": "text", "text": prompt}]
    
    # Add images to the content
    for i, image_path in enumerate(image_paths, 1):
        print(f"  📸 Loading image {i}/{len(image_paths)}: {Path(image_path).name}")
        base64_image = encode_image_to_base64(image_path)
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail
            }
        })
    
    print("\n🚀 Sending request to OpenAI API...")
    
    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=4096
    )
    
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video frames using OpenAI's Vision API."
    )
    parser.add_argument(
        "image_directory",
        type=str,
        help="Directory containing the images to analyze"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="Predict the next 2 plausable actions based on the sequence of images.",
        help="Text prompt for the analysis (default: 'Describe what is happening in this sequence of images. What action is the person performing?')"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o, also available: gpt-4o-mini, gpt-4-turbo)"
    )
    parser.add_argument(
        "-n", "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to analyze (default: all images in directory)"
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: reads from OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "-d", "--detail",
        type=str,
        choices=["low", "high", "auto"],
        default="auto",
        help="Image detail level: 'low' (faster/cheaper), 'high' (more detail), 'auto' (default)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file to save the response (default: print to console only)"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: OpenAI API key not found.")
        print("\n📝 Please provide it via --api-key argument or set OPENAI_API_KEY environment variable.")
        print("\n💡 Example:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   python3 analyze_frames.py /path/to/images --prompt \"Your prompt here\"")
        print("\n🔗 Get your API key at: https://platform.openai.com/api-keys")
        return 1
    
    try:
        # Load images
        image_paths = load_images_from_directory(args.image_directory, args.max_images)
        print(f"\n✅ Found {len(image_paths)} images in '{args.image_directory}'")
        
        # Analyze with OpenAI
        response = analyze_frames_with_openai(
            image_paths,
            args.prompt,
            args.model,
            api_key,
            args.detail
        )
        
        # Display response with pretty formatting
        print("\n" + "━"*80)
        print("📊 ANALYSIS RESULT")
        print("━"*80)
        print(f"\n📁 Directory: {args.image_directory}")
        print(f"🖼️  Images Analyzed: {len(image_paths)}")
        print(f"🤖 Model: {args.model}")
        print(f"💬 Prompt: {args.prompt}")
        print("\n" + "─"*80)
        print("📝 Response:")
        print("─"*80)
        print(f"\n{response}\n")
        print("━"*80)
        
        # Save to file if specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("ANALYSIS RESULT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Directory: {args.image_directory}\n")
                f.write(f"Images Analyzed: {len(image_paths)}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Prompt: {args.prompt}\n\n")
                f.write("-"*80 + "\n")
                f.write("RESPONSE:\n")
                f.write("-"*80 + "\n\n")
                f.write(response)
                f.write("\n\n" + "="*80 + "\n")
            print(f"\n💾 Response saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

