use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, Read},
    path::Path,
};

fn main() -> io::Result<()> {
    let infernal_dir = Path::new("/tmp/infernal.tensors");
    let ggml_dir = Path::new("/tmp/ggml.tensors");

    compare_files(infernal_dir, ggml_dir)?;

    Ok(())
}

fn compare_files(dir_a: &Path, dir_b: &Path) -> io::Result<()> {
    let mut files_a = HashMap::new();
    let mut files_b = HashMap::new();

    // Read directory A and store file paths
    for entry in fs::read_dir(dir_a)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            files_a.insert(path.file_name().unwrap().to_owned(), path);
        }
    }

    // Read directory B and store file paths
    for entry in fs::read_dir(dir_b)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            files_b.insert(path.file_name().unwrap().to_owned(), path);
        }
    }

    // Ensure the keys (filenames) are sorted to maintain execution order
    let mut sorted_keys: Vec<_> = files_a.keys().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        if let Some(path_a) = files_a.get(key) {
            if let Some(path_b) = files_b.get(key) {
                // Compare file contents
                let data_a = read_file_to_vec(path_a)?;
                let data_b = read_file_to_vec(path_b)?;

                if data_a.len() != data_b.len() {
                    println!(
                        "File {key:?} has different sizes: a={} != b={}",
                        data_a.len(),
                        data_b.len()
                    );
                    // Early exit.
                    return Ok(());
                }

                let mismatches = data_a
                    .iter()
                    .zip(data_b.iter())
                    .filter(|&(a, b)| a != b)
                    .count();
                let total = data_b.len();
                if mismatches > 0 {
                    // Mismatch found, print N bytes afterwards for debugging.
                    const N: usize = 100;
                    for (i, (a, b)) in data_a.iter().zip(data_b.iter()).enumerate() {
                        if a != b {
                            println!("Mismatch at byte {i}: {a} != {b}",);
                            for j in i..i + N {
                                if j < data_a.len() {
                                    println!("  {j}: {a} <?> {b}", a = data_a[j], b = data_b[j]);
                                }
                            }
                            break;
                        }
                    }
                }

                if mismatches > 0 {
                    println!("Mismatch in file {key:?}: {mismatches}/{total} bytes differ");
                    // Early exit.
                    return Ok(());
                }
            } else {
                println!("File {key:?} found in {dir_a:?} but not in {dir_b:?}");
            }
        }
    }

    Ok(())
}

fn read_file_to_vec(file_path: &Path) -> io::Result<Vec<u8>> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}
