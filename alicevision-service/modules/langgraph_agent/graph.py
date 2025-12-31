"""
🔗 LangGraph Workflow Definition
Agentic graph for Digital Stylist

Creates cyclic workflow:
Supervisor → Perception/Styling/Simulation → Supervisor → ...
"""

from typing import Dict, List, Any, Optional
import logging
import uuid
from datetime import datetime

from .state import StylistState, UserPreferences
from .supervisor import StylistSupervisor
from .perception_tool import PerceptionTool
from .styling_agent import StylingAgent
from .simulation_tool import SimulationTool

logger = logging.getLogger(__name__)


class DigitalStylistGraph:
    """
    LangGraph-compatible Digital Stylist workflow.
    
    Nodes:
    - supervisor: Routes user intent
    - perception: Analyzes uploaded images
    - styling: Generates recommendations
    - simulation: Runs virtual try-on
    
    Edges:
    - supervisor → (perception | styling | simulation | END)
    - perception → supervisor
    - styling → supervisor
    - simulation → supervisor
    """
    
    def __init__(self):
        """Initialize graph nodes."""
        self.supervisor = StylistSupervisor()
        self.perception = PerceptionTool()
        self.styling = StylingAgent()
        self.simulation = SimulationTool()
        
        # Session storage
        self.sessions: Dict[str, StylistState] = {}
    
    def get_or_create_session(
        self,
        session_id: str = None,
        user_id: str = None
    ) -> StylistState:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        state = StylistState(
            session_id=new_session_id,
            user_preferences=UserPreferences(user_id=user_id or "")
        )
        
        self.sessions[new_session_id] = state
        return state
    
    def run(
        self,
        user_message: str,
        session_id: str = None,
        images: List[str] = None,
        max_iterations: int = 5
    ) -> Dict:
        """
        Run the stylist graph.
        
        Args:
            user_message: User's message
            session_id: Session ID (optional)
            images: List of base64 images (optional)
            max_iterations: Max graph iterations
            
        Returns:
            Response dict with assistant message and state
        """
        # Get session
        state = self.get_or_create_session(session_id)
        
        # Add user message
        state.current_query = user_message
        state.add_message("user", user_message)
        
        # Add pending images
        if images:
            state.pending_uploads = images
        
        # Run graph
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            state.iteration_count = iteration
            
            logger.info(f"Iteration {iteration}: Node = {state.current_node}")
            
            # Route through supervisor
            if state.current_node == "supervisor":
                next_node, state = self.supervisor.route(state)
                
                if next_node == "END" or state.should_end:
                    break
                
                state.current_node = next_node
            
            # Execute node
            elif state.current_node == "perception_tool":
                state = self.perception.process(state)
            
            elif state.current_node == "styling_agent":
                state = self.styling.process(state)
            
            elif state.current_node == "simulation_tool":
                state = self.simulation.process(state)
            
            else:
                logger.warning(f"Unknown node: {state.current_node}")
                break
            
            # Check if we should end
            if state.should_end:
                break
        
        # Save session
        self.sessions[state.session_id] = state
        
        # Get last assistant message
        last_message = ""
        for msg in reversed(state.messages):
            if msg["role"] == "assistant":
                last_message = msg["content"]
                break
        
        if not last_message:
            last_message = self.supervisor.generate_response(state)
        
        return {
            "success": True,
            "sessionId": state.session_id,
            "message": last_message,
            "wardrobeCount": len(state.wardrobe_inventory),
            "recommendationCount": len(state.current_recommendations),
            "iterations": iteration
        }


# Singleton graph
_stylist_graph = None

def get_stylist_graph() -> DigitalStylistGraph:
    """Get singleton stylist graph."""
    global _stylist_graph
    if _stylist_graph is None:
        _stylist_graph = DigitalStylistGraph()
    return _stylist_graph


def create_stylist_graph() -> DigitalStylistGraph:
    """Create new stylist graph."""
    return DigitalStylistGraph()


def run_stylist_conversation(
    message: str,
    session_id: str = None,
    images: List[str] = None
) -> Dict:
    """
    Convenience function to run stylist conversation.
    
    Args:
        message: User message
        session_id: Optional session ID
        images: Optional list of base64 images
        
    Returns:
        Response dict
    """
    graph = get_stylist_graph()
    return graph.run(message, session_id, images)


# ============================================
# LANGGRAPH NATIVE INTEGRATION
# ============================================

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    LANGGRAPH_AVAILABLE = True
    
    def create_langgraph_native():
        """
        Create native LangGraph StateGraph.
        
        Use this for production with proper state persistence.
        """
        from .supervisor import create_supervisor_node
        from .perception_tool import create_perception_node
        from .styling_agent import create_styling_node
        from .simulation_tool import create_simulation_node
        
        # Define graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("supervisor", create_supervisor_node())
        workflow.add_node("perception", create_perception_node())
        workflow.add_node("styling", create_styling_node())
        workflow.add_node("simulation", create_simulation_node())
        
        # Add edges
        workflow.set_entry_point("supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next", "styling"),
            {
                "perception_tool": "perception",
                "styling_agent": "styling",
                "simulation_tool": "simulation",
                "END": END
            }
        )
        
        workflow.add_edge("perception", "supervisor")
        workflow.add_edge("styling", "supervisor")
        workflow.add_edge("simulation", "supervisor")
        
        # Compile with checkpointer
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        
        return workflow.compile(checkpointer=checkpointer)

except ImportError:
    LANGGRAPH_AVAILABLE = False
    
    def create_langgraph_native():
        """LangGraph not available."""
        logger.warning("LangGraph not installed. Using fallback graph.")
        return None
