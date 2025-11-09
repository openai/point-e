# Text2PointCloud Notebook Modifications

## Overview
The `text2pointcloud_test.ipynb` notebook has been modified to generate 100 variations of point clouds with different prompts and model parameters.

## What Changed

### Original Behavior
- Generated single point clouds one at a time
- Manual parameter adjustment required for variations
- No systematic tracking of parameters
- Random filenames without organization

### New Behavior
- Automatically generates 100 variations in a loop
- Systematically varies prompts and model parameters
- Saves all point clouds with sequential IDs (pointcloud_001.ply to pointcloud_100.ply)
- Creates a CSV file tracking all parameters for each variation

## Notebook Structure

### Cells 0-4: Setup (unchanged)
- Cell 0: Clone repository
- Cell 1: Change to point-e directory
- Cell 2: Install dependencies
- Cell 3: Import libraries
- Cell 4: Load models and checkpoints

### Cells 5-8: New Variation Generation
- **Cell 5**: Configuration
  - Defines 15 different text prompts
  - Sets up parameter ranges:
    - `guidance_scale`: 2.0 to 5.0 (7 values)
    - `karras_steps`: 64, 96, 128 (3 values)
    - `s_churn`: 0.5 to 5.0 (5 values)
  - Creates output directory `pointcloud_variations/`

- **Cell 6**: Generation Loop
  - Generates 100 variations
  - Cycles through prompts and parameters
  - Saves each point cloud as `pointcloud_XXX.ply`
  - Records all parameters to tracking data
  - Shows progress for each variation

- **Cell 7**: Save Tracking Data
  - Creates `pointcloud_parameters.csv`
  - Displays first and last 10 variations
  - Shows total count

- **Cell 8**: Optional Visualization
  - Allows viewing parameters for specific variations
  - Change `variation_to_view` to inspect different results

## Output Files

### Point Cloud Files
- Location: `pointcloud_variations/`
- Format: `pointcloud_001.ply` through `pointcloud_100.ply`
- Each file contains 4096 points with RGB color

### Parameter Tracking CSV
- Filename: `pointcloud_parameters.csv`
- Columns:
  - `id`: Variation number (1-100)
  - `prompt`: Text description used
  - `guidance_scale_base`: Guidance scale for base model
  - `guidance_scale_upsample`: Guidance scale for upsampler (always 0.0)
  - `karras_steps_base`: Number of diffusion steps for base model
  - `karras_steps_upsample`: Number of diffusion steps for upsampler
  - `s_churn_base`: Stochasticity parameter for base model
  - `s_churn_upsample`: Stochasticity parameter for upsampler
  - `num_points`: Total points in cloud (always 4096)
  - `filename`: PLY filename
  - `timestamp`: When the variation was generated

## Parameter Variation Strategy

The notebook systematically varies parameters to create diverse outputs:

1. **Prompts** (15 variations): Cycles through different furniture descriptions
2. **Guidance Scale** (7 values): Controls how closely the model follows the prompt
   - Lower values (2.0-3.0): More creative/organic results
   - Higher values (4.0-5.0): More precise adherence to prompt
3. **Karras Steps** (3 values): Number of diffusion steps
   - 64: Faster, less refined
   - 96: Balanced
   - 128: Slower, more refined
4. **S-Churn** (5 values): Controls randomness/noise
   - Lower (0.5-1.0): Cleaner, sharper boundaries
   - Higher (3.0-5.0): More organic, varied appearance

## Usage

1. Run cells 0-4 to set up the environment (one-time setup)
2. Run cell 5 to configure parameters
3. Run cell 6 to generate all 100 variations (this will take time!)
4. Run cell 7 to save and view the tracking data
5. Run cell 8 to inspect parameters of specific variations

## Finding Successful Models

After generation, you can:
1. Open `pointcloud_parameters.csv` in a spreadsheet application
2. Review the point cloud files in `pointcloud_variations/`
3. Identify successful variations by their ID number
4. Look up the exact parameters in the CSV file
5. Use those parameters to generate similar models

## Example Workflow

```python
# After running all cells, to find parameters for variation 42:
params = df[df['id'] == 42].iloc[0]
print(f"Prompt: {params['prompt']}")
print(f"Guidance: {params['guidance_scale_base']}")
print(f"Steps: {params['karras_steps_base']}")
print(f"Churn: {params['s_churn_base']}")
```

## Customization

To modify the variations:
- **Change prompts**: Edit the `prompts` list in cell 5
- **Adjust parameter ranges**: Modify `guidance_scales`, `karras_steps_options`, or `s_churn_options`
- **Generate more/fewer variations**: Change `num_variations` in cell 6
- **Change output directory**: Modify `output_dir` in cell 5
