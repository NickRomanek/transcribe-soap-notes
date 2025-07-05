#!/usr/bin/env python3
"""
Command-line interface for AI Scribe
"""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import ai_scribe.pipeline as pipeline
from ai_scribe.config import OUTPUT_DIR

app = typer.Typer(
    name="ai-scribe",
    help="Offline AI transcription, diarization, and summarization",
    add_completion=False
)
console = Console()


@app.command()
def run(
    wav_path: str = typer.Argument(..., help="Path to WAV audio file"),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--no-pretty", help="Pretty print JSON output"
    )
):
    """Process audio file through complete AI transcription pipeline"""
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(name)s - %(message)s'
    )
    
    try:
        # Validate input
        wav_file = Path(wav_path)
        pipeline.validate_audio_file(wav_file)
        
        output_path = Path(output_dir) if output_dir else OUTPUT_DIR
        
        console.print(f"[green]Processing:[/green] {wav_file}")
        console.print(f"[blue]Output dir:[/blue] {output_path}")
        
        # Run pipeline with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing audio...", total=None)
            
            results = pipeline.run(str(wav_file), output_path)
            progress.remove_task(task)
        
        # Print results
        if pretty:
            console.print("\n[green bold]✓ Processing Complete![/green bold]")
            
            # Print summary table
            table = Table(title="Processing Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            stats = results.get("statistics", {})
            table.add_row("Duration", f"{stats.get('total_duration_minutes', 0):.1f} minutes")
            table.add_row("Speakers", str(stats.get('unique_speakers', 0)))
            table.add_row("Segments", str(stats.get('total_segments', 0)))
            table.add_row("Processing Time", f"{results['metadata']['processing_time_seconds']:.1f}s")
            
            console.print(table)
            
            # Print SOAP summary if available
            soap = results.get("soap_summary", {})
            if "soap_note" in soap:
                console.print("\n[yellow bold]SOAP Note Summary:[/yellow bold]")
                for section, content in soap["soap_note"].items():
                    console.print(f"[cyan]{section.upper()}:[/cyan] {content[:100]}...")
            
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False))
            
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise typer.Exit(1)


@app.command()
def list_results(
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory to search"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show")
):
    """List recent processing results"""
    
    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    results = pipeline.list_recent_results(output_path, limit)
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table(title=f"Recent Results ({output_path})")
    table.add_column("File", style="cyan")
    table.add_column("Input", style="white")
    table.add_column("Duration", justify="right")
    table.add_column("Speakers", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Processed", style="dim")
    
    for result in results:
        status = "[red]Error[/red]" if result["has_error"] else "[green]✓[/green]"
        table.add_row(
            result["filename"],
            Path(result["input_file"]).name,
            f"{result['duration_minutes']:.1f}m",
            str(result["speakers"]),
            status,
            result["timestamp"][:16]  # Show date/time only
        )
    
    console.print(table)


@app.command()
def validate(
    wav_path: str = typer.Argument(..., help="Path to audio file to validate")
):
    """Validate audio file for processing"""
    
    try:
        wav_file = Path(wav_path)
        pipeline.validate_audio_file(wav_file)
        
        # Get file info
        file_size_mb = wav_file.stat().st_size / (1024 * 1024)
        
        console.print(f"[green]✓ Valid audio file[/green]")
        console.print(f"File: {wav_file}")
        console.print(f"Size: {file_size_mb:.1f} MB")
        console.print(f"Format: {wav_file.suffix}")
        
    except Exception as e:
        console.print(f"[red]✗ Invalid audio file:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 