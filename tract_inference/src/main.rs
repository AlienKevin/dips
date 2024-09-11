use std::{
    path::{Path, PathBuf},
    str::FromStr,
};
use tokenizers::tokenizer::{Result, Tokenizer};
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    let model_dir = PathBuf::from_str("./model")?;
    let tokenizer = Tokenizer::from_file(Path::join(&model_dir, "tokenizer.json"))?;

    let text = "9月嘅天氣唔錯";

    let tokenizer_output = tokenizer.encode(text, true)?;
    let input_ids = tokenizer_output.get_ids();
    let attention_mask = tokenizer_output.get_attention_mask();
    let token_type_ids = tokenizer_output.get_type_ids();
    let length = input_ids.len();

    let model = tract_onnx::onnx()
        .model_for_path(Path::join(&model_dir, "model.onnx"))?
        .into_optimized()?
        .into_runnable()?;
    println!("{:?}", input_ids);
    let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        input_ids.iter().map(|&x| x as i64).collect(),
    )?
    .into();
    let attention_mask: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    )?
    .into();
    let token_type_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        token_type_ids.iter().map(|&x| x as i64).collect(),
    )?
    .into();

    let outputs =
        model.run(tvec!(input_ids.into(), attention_mask.into(), token_type_ids.into()))?;
    println!("{:?}", outputs);
    let logits = outputs[0].to_array_view::<f16>()?;
    println!("{:?}", logits);
    
    let mut result = String::new();
    for row in logits.rows() {
        let max_index = row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);
        
        let label = match max_index {
            0 => "D",
            1 => "I",
            2 => "P",
            3 => "S",
            _ => panic!("unknown label index {}", max_index)
        };
        result.push_str(label);
    }
    result = result[1..result.len() - 1].to_string();
    println!("Input:    {}", text);
    println!("Predicted:{}", result);

    Ok(())
}
