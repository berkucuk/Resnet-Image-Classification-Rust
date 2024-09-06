use anyhow::Result;
use tch::{
    Device,
    Kind,
    nn::VarStore,
    vision::{
        imagenet,
        resnet::resnet18,
    }
};

fn main() -> Result<()> {

    // Create the model and load the pre-trained weights
   let mut vs = VarStore::new(Device::cuda_if_available());
   let model = resnet18(&vs.root(), 1000);
  
  
   vs.load("src/model.safetensors")?;
   
   let image = imagenet::load_image_and_resize224("src/1.jpg")?
    .to_device(vs.device());
  
   let output = image
    .unsqueeze(0)
    .apply_t(&model, false)
    .softmax(-1, Kind::Float);
   
      for (probability, class) in imagenet::top(&output, 5).iter() {
          println!("{:50} {:5.2}%", class, 100.0 * probability)
      }
      Ok(())
  
  }
