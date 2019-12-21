# Steel_classifier

I tried to use transfer learning from a VGG16 trained on imagenet to try to classify Steel surface defects. Since imagenet is mostly a consumer photos type database, I thought that I woudld get terrible results but was pleasanly surprised with that i got 

The Steel dataset is public and can be accessed here http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

If you are curious, the best way to follow along is to open up the notebook titled 'Training_and_inference_notebook' and follow along the steps there 

Here are a few things I learned as I did this exercise - 
  #1 Transfer learning is very powerful 
  #2 In most cases, you don't need powerful GPUs to train if you used the power of transfer learning. My entire code trained  within 5 minutes 
  

If you use any parts of the code, a reference back to this github page will be appreciated 
