use anyhow::Result;

fn main() -> Result<()> {
    println!("Starting Ferrotensor Monitor TUI Demo...");
    // Call the library run function
    ferrotensor_monitor::run()?;
    Ok(())
}
