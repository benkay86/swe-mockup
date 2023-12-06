//! Write mock data to a series of npy files in the current directory.

use std::fs::File;
use std::io::Write;
use swe_mockup::{MockData, MockParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Mock data parmeters.
    let params = MockParams::default();
    print!("{}", params);

    // Generate mock data.
    print!("Generating mock data...");
    std::io::stdout().flush().unwrap();
    let mock_data = MockData::<f64>::from_params(params);
    println!(" done.");
    println!("Generated {} blocks.", mock_data.n_blocks);

    // Write data to npz file.
    print!("Writing mock data to mock-data.npz...");
    std::io::stdout().flush().unwrap();
    mock_data.save_npz_file(File::create("mock-data.npz")?)?;
    println!(" done.");

    Ok(())
}