use anyhow::Result;

fn main() -> Result<()> {
    println!("Starting Rusty Monitor TUI Demo...");
    // Call the library run function
    rusty_monitor::run()?;
    Ok(())
}
