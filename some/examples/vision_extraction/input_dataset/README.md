# Product Images Dataset

This directory contains sample product images for testing multimodal extraction capabilities.

**Note**: The sample product images referenced in this example were generated using [FLUX.1-dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev), a state-of-the-art text-to-image generation model by Black Forest Labs.

## Directory Structure

```
input_dataset/
├── images/                    # Product image files
│   ├── smartphone_box.jpg     # Electronics example
│   ├── coffee_bag.jpg         # Food/beverage example  
│   ├── running_shoes.jpg      # Sports/apparel example
│   ├── skincare_bottle.jpg    # Beauty/cosmetics example
│   └── book_cover.jpg         # Books/media example
├── sample_products.json       # Sample data with expected results
└── README.md                  # This file
```

## Sample Data Format

The `sample_products.json` file contains test cases with the following structure:

```json
{
  "id": "unique_product_id",
  "image_path": "relative/path/to/image.jpg",
  "additional_text": "Optional context or description",
  "expected_details": {
    "name": "Expected product name",
    "brand": "Expected brand",
    "category": "Expected category",
    "key_features": ["Expected", "features"]
  }
}
```

## Adding Your Own Images

To test with your own product images:

1. Add image files to the `images/` directory
2. Update `sample_products.json` with corresponding entries
3. Supported image formats: JPG, PNG, WebP, GIF
4. Recommended image size: 1024x1024 pixels or smaller for optimal processing

## Image Guidelines

For best extraction results, use images that:

- **Clear and well-lit**: Product details should be easily visible
- **High resolution**: Text and labels should be readable
- **Proper framing**: Product should fill most of the frame
- **Minimal background**: Focus on the product itself
- **Multiple angles**: Include packaging, labels, and product views

## Expected Categories

The extraction system recognizes these product categories:

- `electronics` - Phones, computers, gadgets
- `clothing` - Apparel, accessories, footwear  
- `food` - Food items, beverages, snacks
- `books` - Books, magazines, media
- `home` - Furniture, decor, household items
- `sports` - Athletic gear, fitness equipment
- `beauty` - Cosmetics, skincare, personal care
- `automotive` - Car parts, accessories
- `other` - Miscellaneous products

## Testing Notes

- The extraction system works best with clear product packaging
- Brand logos and text significantly improve accuracy
- Price tags and labels are automatically detected when visible
- Confidence scores reflect image quality and extraction certainty
