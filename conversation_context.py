"""
Intelligent conversation context management
Handles dialog history and maintains conversation state
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import structlog

logger = structlog.get_logger()

@dataclass
class Turn:
    """Single conversation turn"""
    user_text: str
    bot_response: str
    timestamp: float
    confidence: float = 1.0

class ConversationContext:
    """Manages conversation state and history"""

    def __init__(self, max_turns: int = 4, session_timeout: float = 300.0):
        self.max_turns = max_turns
        self.session_timeout = session_timeout
        self.turns: List[Turn] = []
        self.session_start = time.time()
        self.greeted = False
        self.escalation_count = 0
        self.user_frustration_level = 0

    def add_turn(self, user_text: str, bot_response: str, confidence: float = 1.0):
        """Add a new conversation turn"""
        turn = Turn(
            user_text=user_text.strip(),
            bot_response=bot_response.strip(),
            timestamp=time.time(),
            confidence=confidence
        )

        self.turns.append(turn)

        # Keep only recent turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        # Update conversation state
        self._update_conversation_state(user_text)

        logger.info("Turn added to context",
                   user_text=user_text[:50],
                   bot_response=bot_response[:50],
                   total_turns=len(self.turns))

    def _update_conversation_state(self, user_text: str):
        """Update conversation state based on user input"""
        user_lower = user_text.lower()

        # Detect frustration keywords
        frustration_keywords = [
            "frustrat", "annoying", "stupid", "useless", "terrible",
            "horrible", "awful", "waste", "hate", "angry"
        ]

        if any(keyword in user_lower for keyword in frustration_keywords):
            self.user_frustration_level += 1
            logger.warning("User frustration detected", level=self.user_frustration_level)

        # Detect escalation requests
        escalation_keywords = [
            "human", "person", "agent", "representative", "manager",
            "transfer", "escalate", "speak to someone", "real person"
        ]

        if any(keyword in user_lower for keyword in escalation_keywords):
            self.escalation_count += 1
            logger.warning("Escalation request detected", count=self.escalation_count)

    def build_prompt(self, current_input: str, system_prompt: str) -> str:
        """Build context-aware prompt for LLM"""

        # Start with system prompt
        prompt_parts = [system_prompt]

        # Add conversation history (last few turns)
        if self.turns:
            prompt_parts.append("\nRecent conversation:")
            for turn in self.turns[-2:]:  # Last 2 turns for context
                prompt_parts.append(f"Human: {turn.user_text}")
                prompt_parts.append(f"Assistant: {turn.bot_response}")

        # Add current input
        prompt_parts.append(f"\nHuman: {current_input}")
        prompt_parts.append("Assistant:")

        full_prompt = "\n".join(prompt_parts)

        logger.debug("Prompt built",
                    turns_included=len(self.turns),
                    prompt_length=len(full_prompt),
                    greeted=self.greeted)

        return full_prompt

    def should_escalate(self) -> bool:
        """Determine if conversation should be escalated to human"""
        # Escalate if user explicitly requested multiple times
        if self.escalation_count >= 2:
            return True

        # Escalate if user is very frustrated
        if self.user_frustration_level >= 3:
            return True

        # Escalate if session is too long
        session_duration = time.time() - self.session_start
        if session_duration > self.session_timeout:
            return True

        return False

    def get_greeting_prompt(self) -> str:
        """Get appropriate greeting based on conversation state"""
        if not self.greeted:
            self.greeted = True
            return "Hello, this is Alexis from NETOVO. How can I help you today?"

        # Return empty string - no re-greeting needed
        return ""

    def reset_session(self):
        """Reset conversation for new session"""
        self.turns.clear()
        self.session_start = time.time()
        self.greeted = False
        self.escalation_count = 0
        self.user_frustration_level = 0
        logger.info("Conversation context reset")

    def get_summary(self) -> Dict:
        """Get conversation summary for monitoring"""
        return {
            "total_turns": len(self.turns),
            "session_duration": time.time() - self.session_start,
            "greeted": self.greeted,
            "escalation_count": self.escalation_count,
            "frustration_level": self.user_frustration_level,
            "should_escalate": self.should_escalate()
        }