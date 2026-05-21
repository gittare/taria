use std::env;
use std::fs;
use std::process;

use frontend::parser::Parser;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file.taria>", args[0]);
        process::exit(1);
    }

    let filename = &args[1];
    let source = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file {}: {}", filename, e);
            process::exit(1);
        }
    };

    println!("Parsing {}...", filename);
    let mut parser = Parser::new(&source);
    let ast = parser.parse_module();

    println!("AST Parsed successfully. Found {} functions.", ast.functions.len());
    for func in ast.functions {
        println!(" - Function: {}", func.name);
        for dec in func.decorators {
            println!("    @{}", dec.name);
        }
    }
}
