import time
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual.worker import Worker

# Import de ta logique backend existante
import config as cfg
import cogitator as cog

class CogitatorTUI(App):
    CSS = """
    Screen {
        background: #030803;
        color: #00ff66;
    }
    
    Header {
        background: #051a05;
        color: #00ff66;
        border-bottom: heavy #00ff66;
    }

    Footer {
        background: #051a05;
        color: #00ff66;
    }

    /* Main Container */
    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    /* Level 1: Dialogue Panel */
    #dialogue-panel {
        width: 2fr;
        border: solid #00ff66;
        padding: 1;
        margin: 1;
        background: #050f05;
    }

    /* Level 2: System Logs */
    #logs-panel {
        width: 1fr;
        border: dashed #005522;
        padding: 1;
        margin: 1;
        background: #020502;
    }

    .box-title {
        color: #ffff00;
        text-style: bold;
        margin-bottom: 1;
    }

    /* Bottom Text Input */
    Input {
        dock: bottom;
        background: #050f05;
        color: #00ff66;
        border: solid #00ff66;
        margin: 1;
    }
    
    #chat-log {
        height: 1fr;
        color: #00ff66;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="dialogue-panel"):
                yield Static("=== COMMAND CHANNEL (LEVEL 1) ===", classes="box-title")
                yield RichLog(id="chat-log", highlight=True, markup=True)
            with Vertical(id="logs-panel"):
                yield Static("=== SYSTEM LOGS (LEVEL 2) ===", classes="box-title")
                yield RichLog(id="system-logs", highlight=True, markup=True)
        yield Input(placeholder="Type a text query or use the microphone...")
        yield Footer()

    def on_mount(self) -> None:
        """Executed on startup: initializes engines and starts background mic loop."""
        chat_log = self.query_one("#chat-log", RichLog)
        sys_logs = self.query_one("#system-logs", RichLog)
        
        sys_logs.write("[INIT] Loading Cogitator engines...")
        chat_log.write("[bold red][SERVITOR][/bold red] Cogitator online. Praise the Omnissiah.")
        
        # Startup voice synthesis
        cog.speak_servitor("Cogitator online. State designated query.")

        # Launch background voice listening loop (Worker thread)
        self.run_worker(self.background_voice_loop, thread=True)

    def background_voice_loop(self) -> Worker:
        """Background voice loop (VAD + Whisper) running safely without blocking the UI."""
        chat_log = self.query_one("#chat-log", RichLog)
        sys_logs = self.query_one("#system-logs", RichLog)

        # Phonetic variations for shutdown command tolerance
        shutdown_phrases = ["exit", "terminate", "quit", "stop", "shut down", "should done", "shoot down", "shut one"]

        while True:
            try:
                sys_logs.write("[MIC] Listening for voice trigger (VAD)...")
                audio_file = cog.record_audio_with_vad()
                
                if not audio_file:
                    continue

                sys_logs.write("[STT] Transcribing audio...")
                user_input = cog.transcribe_audio(audio_file)

                if not user_input:
                    continue

                # Display Level 1 (User Vox)
                chat_log.write(f"\n[bold white]> USER [VOX] :[/bold white] {user_input}")
                sys_logs.write(f"[USER] {user_input}")

                # Check for phonetic shutdown command
                if any(phrase in user_input.lower() for phrase in shutdown_phrases):
                    chat_log.write("[bold red]> SERVITOR :[/bold red] Terminating cognitive session.")
                    sys_logs.write("[SHUTDOWN] Exit command recognized via voice.")
                    cog.speak_servitor("Terminating cognitive session.")
                    self.exit()
                    break

                # LLM + RAG Processing
                sys_logs.write("[LLM] Querying cognitive core...")
                response_text = cog.query_llm(user_input)

                # Display Level 1 (Servitor Response)
                chat_log.write(f"[bold green]> SERVITOR :[/bold green] {response_text}")
                
                # Voice Synthesis & DSP
                sys_logs.write("[TTS] Generating speech and applying DSP effects...")
                cog.speak_servitor(response_text)

            except Exception as e:
                sys_logs.write(f"[ERROR] {e}")
                time.sleep(1)

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handles direct keyboard text input in the prompt line."""
        user_input = message.value
        if not user_input:
            return
        
        message.input.value = "" # Clear input box
        chat_log = self.query_one("#chat-log", RichLog)
        sys_logs = self.query_one("#system-logs", RichLog)

        chat_log.write(f"\n[bold white]> USER [TEXT] :[/bold white] {user_input}")
        sys_logs.write(f"[TEXT INPUT] {user_input}")

        # Shutdown command check for text input
        shutdown_phrases = ["exit", "terminate", "quit", "stop", "shut down"]
        if any(p in user_input.lower() for p in shutdown_phrases):
            chat_log.write("[bold red]> SERVITOR :[/bold red] Terminating session.")
            cog.speak_servitor("Terminating session.")
            self.exit()
            return

        # LLM Request
        sys_logs.write("[LLM] Querying cognitive core (Text)...")
        response_text = cog.query_llm(user_input)
        
        chat_log.write(f"[bold green]> SERVITOR :[/bold green] {response_text}")
        
        # Voice Synthesis
        sys_logs.write("[TTS] Speaking response...")
        cog.speak_servitor(response_text)


if __name__ == "__main__":
    app = CogitatorTUI()
    app.run()