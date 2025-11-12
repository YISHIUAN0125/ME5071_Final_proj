from diffusers import DiffusionPipeline
 
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
print(pipeline("An image of a squirrel in Picasso style"))