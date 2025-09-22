#!/usr/bin/env python3
"""
NETOVO VoiceBot - Enhanced Conversation Context Management System
Prevents repetitive greetings and maintains conversation state across turns
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: float
    user_input: str
    bot_response: str
    confidence: float = 0.0
    context_summary: str = ""


class ConversationContextManager:
    """
    Manages conversation context to prevent repetitive greetings and maintain
    professional dialogue flow across multiple turns within a single call.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: List[ConversationTurn] = []
        self.session_start_time = time.time()
        self.first_interaction = True
        self.customer_name = None
        self.customer_issue_type = None
        self.escalation_requested = False
        self.conversation_summary = ""

        # Conversation limits for professional call management
        self.max_turns = 8  # Maximum conversation exchanges
        self.max_call_duration = 300  # 5 minutes max call time
        self.context_window = 3  # Keep last 3 turns for immediate context

    def add_turn(self, user_input: str, bot_response: str, confidence: float = 0.0):
        """Add a new conversation turn and update context"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input.strip(),
            bot_response=bot_response.strip(),
            confidence=confidence
        )

        # Extract key information from this turn
        self._extract_customer_info(user_input)

        # Add to history
        self.conversation_history.append(turn)

        # Mark that first interaction has occurred
        if self.first_interaction:
            self.first_interaction = False

        # Update conversation summary
        self._update_conversation_summary()

        # Trim history if too long (keep recent context + summary)
        self._trim_conversation_history()

    def _extract_customer_info(self, user_input: str):
        """Extract customer information from input"""
        user_lower = user_input.lower()

        # Extract name patterns
        if "my name is" in user_lower or "i'm" in user_lower or "this is" in user_lower:
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ["is", "i'm", "im"] and i + 1 < len(words):
                    potential_name = words[i + 1].strip(".,!?")
                    if potential_name.isalpha() and len(potential_name) > 1:
                        self.customer_name = potential_name
                        break

        # Detect issue types for context
        issue_keywords = {
            "password": "password_reset",
            "email": "email_issue",
            "network": "network_problem",
            "computer": "hardware_issue",
            "software": "software_issue",
            "login": "login_problem",
            "account": "account_access"
        }

        for keyword, issue_type in issue_keywords.items():
            if keyword in user_lower:
                self.customer_issue_type = issue_type
                break

        # Detect escalation requests
        escalation_phrases = ["human", "agent", "person", "supervisor", "manager", "representative"]
        if any(phrase in user_lower for phrase in escalation_phrases):
            self.escalation_requested = True

    def _update_conversation_summary(self):
        """Create a concise summary of the conversation so far"""
        if not self.conversation_history:
            return

        recent_turns = self.conversation_history[-2:]  # Last 2 turns
        issues_mentioned = []

        for turn in recent_turns:
            user_input = turn.user_input.lower()
            if "password" in user_input:
                issues_mentioned.append("password issues")
            elif "email" in user_input:
                issues_mentioned.append("email problems")
            elif "network" in user_input:
                issues_mentioned.append("network connectivity")
            elif "computer" in user_input or "laptop" in user_input:
                issues_mentioned.append("computer problems")

        summary_parts = []
        if self.customer_name:
            summary_parts.append(f"Customer: {self.customer_name}")
        if issues_mentioned:
            summary_parts.append(f"Discussing: {', '.join(set(issues_mentioned))}")
        if self.escalation_requested:
            summary_parts.append("Escalation requested")

        self.conversation_summary = " | ".join(summary_parts)

    def _trim_conversation_history(self):
        """Keep only recent turns plus create summary of older ones"""
        if len(self.conversation_history) > self.context_window:
            # Keep recent turns
            recent_turns = self.conversation_history[-self.context_window:]
            self.conversation_history = recent_turns

    def get_system_prompt(self) -> str:
        """Generate system prompt with current conversation context"""
        base_prompt = """You are Alexis, a professional AI assistant for NETOVO IT Services.

IMPORTANT CONTEXT RULES:
- This is a CONTINUING conversation (not the start of a new call)
- Do NOT repeat your introduction/greeting unless this is genuinely the first exchange
- Maintain professional, helpful tone throughout the conversation
- Focus on the customer's current question or issue"""

        if self.conversation_summary:
            base_prompt += f"\n\nCONVERSATION CONTEXT: {self.conversation_summary}"

        if len(self.conversation_history) > 0:
            base_prompt += f"\nTURN NUMBER: {len(self.conversation_history) + 1}"

            # Add recent conversation context
            if len(self.conversation_history) >= 1:
                last_turn = self.conversation_history[-1]
                base_prompt += f"\nLAST CUSTOMER MESSAGE: {last_turn.user_input}"
                base_prompt += f"\nYOUR LAST RESPONSE: {last_turn.bot_response}"

        if self.escalation_requested:
            base_prompt += "\n\nNOTE: Customer has requested human assistance. Acknowledge this and offer to transfer."

        return base_prompt

    def should_greet(self) -> bool:
        """Determine if bot should give a greeting (only for first interaction)"""
        return self.first_interaction

    def should_escalate(self) -> bool:
        """Check if call should be escalated to human"""
        # Escalate if explicitly requested
        if self.escalation_requested:
            return True

        # Escalate if conversation is too long
        if len(self.conversation_history) >= self.max_turns:
            return True

        # Escalate if call duration too long
        if time.time() - self.session_start_time > self.max_call_duration:
            return True

        return False

    def get_escalation_reason(self) -> str:
        """Get reason for escalation"""
        if self.escalation_requested:
            return "customer_request"
        elif len(self.conversation_history) >= self.max_turns:
            return "conversation_length"
        elif time.time() - self.session_start_time > self.max_call_duration:
            return "call_duration"
        else:
            return "unknown"

    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context for LLM"""
        return {
            "session_id": self.session_id,
            "turn_count": len(self.conversation_history),
            "first_interaction": self.first_interaction,
            "customer_name": self.customer_name,
            "issue_type": self.customer_issue_type,
            "escalation_requested": self.escalation_requested,
            "conversation_summary": self.conversation_summary,
            "call_duration": int(time.time() - self.session_start_time),
            "recent_turns": [asdict(turn) for turn in self.conversation_history[-2:]]
        }

    def format_prompt_for_llm(self, current_user_input: str) -> str:
        """Format complete prompt for LLM including context"""
        system_prompt = self.get_system_prompt()

        # Build conversation context
        context_lines = []

        if self.should_greet():
            context_lines.append("INSTRUCTION: Provide a professional greeting as this is the start of the call.")
        else:
            context_lines.append("INSTRUCTION: Continue the ongoing conversation. DO NOT repeat greetings.")

        # Add recent conversation for context (but not too much)
        if self.conversation_history:
            context_lines.append("\nRECENT CONVERSATION:")
            for turn in self.conversation_history[-2:]:  # Only last 2 turns
                context_lines.append(f"Customer: {turn.user_input}")
                context_lines.append(f"You: {turn.bot_response}")

        context_lines.append(f"\nCURRENT CUSTOMER MESSAGE: {current_user_input}")
        context_lines.append("\nYOUR RESPONSE:")

        return system_prompt + "\n" + "\n".join(context_lines)

    def save_session(self, filepath: str):
        """Save conversation session to file"""
        session_data = {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "conversation_history": [asdict(turn) for turn in self.conversation_history],
            "customer_name": self.customer_name,
            "customer_issue_type": self.customer_issue_type,
            "escalation_requested": self.escalation_requested,
            "conversation_summary": self.conversation_summary
        }

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

    @classmethod
    def load_session(cls, filepath: str, session_id: str) -> 'ConversationContextManager':
        """Load conversation session from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        manager = cls(session_id)
        manager.session_start_time = data.get("start_time", time.time())
        manager.customer_name = data.get("customer_name")
        manager.customer_issue_type = data.get("customer_issue_type")
        manager.escalation_requested = data.get("escalation_requested", False)
        manager.conversation_summary = data.get("conversation_summary", "")

        # Rebuild conversation history
        for turn_data in data.get("conversation_history", []):
            turn = ConversationTurn(**turn_data)
            manager.conversation_history.append(turn)

        manager.first_interaction = len(manager.conversation_history) == 0

        return manager


# Session management for concurrent calls
class SessionManager:
    """Manages multiple conversation sessions for concurrent calls"""

    def __init__(self):
        self.sessions: Dict[str, ConversationContextManager] = {}
        self.session_timeout = 1800  # 30 minutes

    def get_or_create_session(self, session_id: str) -> ConversationContextManager:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContextManager(session_id)
        return self.sessions[session_id]

    def end_session(self, session_id: str):
        """End and cleanup session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup_expired_sessions(self):
        """Remove expired sessions to prevent memory leaks"""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if current_time - session.session_start_time > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.end_session(session_id)


# Global session manager instance
session_manager = SessionManager()


if __name__ == "__main__":
    # Test the conversation context system
    session_id = "test_call_001"
    context = ConversationContextManager(session_id)

    # Simulate conversation
    print("=== Testing Conversation Context ===")

    # First turn - should greet
    print(f"Should greet: {context.should_greet()}")
    user1 = "Hello, I need help with my password"
    bot1 = "Hello! I'm Alexis from NETOVO. I'd be happy to help you with your password issue."
    context.add_turn(user1, bot1, 0.95)
    print(f"After turn 1 - Should greet: {context.should_greet()}")

    # Second turn - should NOT greet
    user2 = "I can't log into my email account"
    prompt = context.format_prompt_for_llm(user2)
    print(f"\nGenerated prompt:\n{prompt}")

    bot2 = "I can help you reset your email password. What email provider are you using?"
    context.add_turn(user2, bot2, 0.92)

    # Check context
    print(f"\nConversation summary: {context.conversation_summary}")
    print(f"Customer name: {context.customer_name}")
    print(f"Issue type: {context.customer_issue_type}")

    print("\n=== Context Test Complete ===")
