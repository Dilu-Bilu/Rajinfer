## What worked? 
Well the first thing that worked was that I used typunit16 for bf16_t. 
-- I also made a new function for matmuls with bf16 since it was not recognized as a template 
-- Then I had to meticulously change everything correctly such that matmuls use their respective quantization 
-- Turned out I made a cuda reference inside of model.cpp for bf16 
-- Temperature was also too high causing me to bug out 
-- Finally, the sampling final embedding was in fp16 and had to change it to bf16 and bf16matmul 