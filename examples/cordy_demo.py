#!/usr/bin/env python3
"""
Updated Cordyceps Example showing guards for selective firing and IO callbacks.

This example demonstrates:
1. Using guard() method for selective token consumption
2. External IO handling via output place callbacks
3. Timer interface with proper token-based cancellation
"""

from cordyceps import Token, Place, ArcBasedTransition, Interface, PetriNet, Arc
import threading
import time


# Token definitions
class TimerRequestToken(Token):
    def __init__(self, timer_id: str, duration: float):
        super().__init__()
        self.timer_id = timer_id
        self.duration = duration


class ActiveTimerToken(Token):
    def __init__(self, timer_id: str, expires_at: float):
        super().__init__()
        self.timer_id = timer_id
        self.expires_at = expires_at


class CancelRequestToken(Token):
    def __init__(self, timer_id: str):
        super().__init__()
        self.timer_id = timer_id


class TimerCompletedToken(Token):
    def __init__(self, timer_id: str):
        super().__init__()
        self.timer_id = timer_id


class TimerCancelledToken(Token):
    def __init__(self, timer_id: str):
        super().__init__()
        self.timer_id = timer_id


class NetworkPacketToken(Token):
    def __init__(self, destination: str, data: str):
        super().__init__()
        self.destination = destination
        self.packet_data = data


class CommandToken(Token):
    def __init__(self, command_type: str, params: dict):
        super().__init__()
        self.command_type = command_type
        self.params = params


# Timer Interface with selective cancellation
class TimerInterface(Interface):
    class RequestPlace(Place): pass
    class ActivePlace(Place): pass
    class CancelPlace(Place): pass
    class CompletedPlace(Place): pass
    class CancelledPlace(Place): pass
    class ExternalTimerPlace(Place): pass  # IO boundary
    
    class StartTimerTransition(ArcBasedTransition):
        def input_arcs(self):
            return {"request": Arc(TimerInterface.RequestPlace)}
        
        def output_arcs(self):
            return {"active": Arc(TimerInterface.ActivePlace), "external": Arc(TimerInterface.ExternalTimerPlace)}
        
        def fire(self, ctx, consumed, produce):
            request = consumed["request"][0]
            
            # Create active timer
            expires_at = time.time() + request.duration
            active = ActiveTimerToken(request.timer_id, expires_at)
            produce.key["active"]([active])
            
            # Send to external timer system
            produce.key["external"]([request])
    
    class CancelTimerTransition(ArcBasedTransition):
        def input_arcs(self):
            return {"active": Arc(TimerInterface.ActivePlace), "cancel": Arc(TimerInterface.CancelPlace)}
        
        def output_arcs(self):
            return {"cancelled": Arc(TimerInterface.CancelledPlace)}
        
        def guard(self, ctx):
            # Use selective consumption to find matching timer/cancel pairs
            try:
                timer, cancel = self.consume_matching_pair(
                    "active", "cancel",
                    lambda t, c: t.timer_id == c.timer_id
                )
                return True
            except:
                return False
        
        def fire(self, ctx, consumed, produce):
            timer = consumed["active"][0]
            cancel = consumed["cancel"][0]
            
            cancelled = TimerCancelledToken(timer.timer_id)
            produce.key["cancelled"]([cancelled])
    
    class TimeoutTransition(ArcBasedTransition):
        def input_arcs(self):
            return {"active": Arc(TimerInterface.ActivePlace)}
        
        def output_arcs(self):
            return {"active": Arc(TimerInterface.ActivePlace), "completed": Arc(TimerInterface.CompletedPlace)}
        
        def guard(self, ctx):
            # Check if any active timers have expired
            current_time = time.time()
            
            for timer in self.peek("active"):
                if current_time >= timer.expires_at:
                    self.consume("active", timer)
                    return True
            return False
        
        def fire(self, ctx, consumed, produce):
            expired_timer = consumed["active"][0]
            completed = TimerCompletedToken(expired_timer.timer_id)
            produce.key["completed"]([completed])


# Network Interface for external communication
class NetworkInterface(Interface):
    class OutputPlace(Place): pass  # IO boundary
    
    class SendMessageTransition(ArcBasedTransition):
        def input_arcs(self):
            return [Arc(Place)]  # Will be fixed by parent system
        
        def output_arcs(self):
            return [Arc(NetworkInterface.OutputPlace)]
        
        def guard(self, ctx):
            # Only fire for send_message commands
            if not self.peek("0"):  # No tokens available
                return False
            
            command = self.peek("0")[0]
            if hasattr(command, 'command_type') and command.command_type == "send_message":
                self.consume("0", command)
                return True
            return False
        
        def fire(self, ctx, consumed, produce):
            command = consumed["0"][0]  # Using index access for list format
            
            packet = NetworkPacketToken(
                command.params["destination"],
                command.params["message"]
            )
            produce.index[0]([packet])


# Main Robot System
class RobotSystem(PetriNet):
    class Timer(TimerInterface): pass
    class Network(NetworkInterface): pass
    
    class CommandPlace(Place): pass
    class StatusPlace(Place): pass
    
    class ProcessCommandTransition(ArcBasedTransition):
        def input_arcs(self):
            return {"command": Arc(RobotSystem.CommandPlace)}
        
        def output_arcs(self):
            return {
                "timer_request": Arc(RobotSystem.Timer.RequestPlace),
                "cancel_request": Arc(RobotSystem.Timer.CancelPlace),
                "status": Arc(RobotSystem.StatusPlace)
            }
        
        def fire(self, ctx, consumed, produce):
            command = consumed["command"][0]
            
            if command.command_type == "start_timer":
                timer_req = TimerRequestToken(
                    command.params["timer_id"],
                    command.params["duration"]
                )
                produce.key["timer_request"]([timer_req])
                
            elif command.command_type == "cancel_timer":
                cancel_req = CancelRequestToken(command.params["timer_id"])
                produce.key["cancel_request"]([cancel_req])
                
            elif command.command_type == "send_message":
                # For network messages, let the NetworkInterface handle them
                # We don't route them back to CommandPlace to avoid loops
                pass
            
            # Always update status
            status = Token(f"Processed: {command.command_type}")
            produce.key["status"]([status])
    
    def build(self):
        """Custom build to fix NetworkInterface connections."""
        # Fix the NetworkInterface SendMessageTransition to connect to our CommandPlace
        for transition in self._transitions:
            if isinstance(transition, self.Network.SendMessageTransition):
                # Override the input_arcs method
                transition.input_arcs = lambda: [Arc(RobotSystem.CommandPlace)]


# External IO Handlers
class ExternalTimerManager:
    def __init__(self, robot_net):
        self.robot_net = robot_net
        self.active_timers = {}
    
    def handle_timer_request(self, token, place):
        """Called when tokens are added to ExternalTimerPlace."""
        if isinstance(token, TimerRequestToken):
            print(f"Starting external timer: {token.timer_id} for {token.duration}s")
            
            # Start actual timer
            timer = threading.Timer(
                token.duration,
                lambda: self._timer_expired(token.timer_id)
            )
            timer.start()
            self.active_timers[token.timer_id] = timer
    
    def _timer_expired(self, timer_id):
        """Called when external timer expires."""
        print(f"Timer {timer_id} expired!")
        # Send timeout event back to the net
        # (In real implementation, this would trigger the TimeoutTransition)
        if timer_id in self.active_timers:
            del self.active_timers[timer_id]


class NetworkManager:
    def __init__(self):
        self.sent_packets = []
    
    def handle_network_output(self, token, place):
        """Called when tokens are added to NetworkOutputPlace."""
        if isinstance(token, NetworkPacketToken):
            print(f"Sending packet to {token.destination}: {token.packet_data}")
            self.sent_packets.append((token.destination, token.packet_data))
            # In real implementation: send via actual network


def demonstrate_system():
    """Demonstrates the updated Cordyceps system."""
    print("=== Updated Cordyceps: Guards and IO Example ===\n")
    
    # Create the robot system
    robot = RobotSystem()
    
    # Set up external IO handlers
    timer_manager = ExternalTimerManager(robot)
    network_manager = NetworkManager()
    
    robot.attach_output_handler(
        RobotSystem.Timer.ExternalTimerPlace,
        timer_manager.handle_timer_request
    )
    robot.attach_output_handler(
        RobotSystem.Network.OutputPlace,
        network_manager.handle_network_output
    )
    
    print("System initialized with IO handlers")
    print()
    
    # Test timer operations
    print("=== Testing Timer Operations ===")
    
    # Start a timer
    robot.send_message_to_place(
        RobotSystem.CommandPlace,
        CommandToken("start_timer", {"timer_id": "task1", "duration": 2.0})
    )
    robot.tick()
    print(f"Active timers: {robot.context.token_count(RobotSystem.Timer.ActivePlace)}")
    
    # Start another timer
    robot.send_message_to_place(
        RobotSystem.CommandPlace,
        CommandToken("start_timer", {"timer_id": "task2", "duration": 3.0})
    )
    robot.tick()
    print(f"Active timers: {robot.context.token_count(RobotSystem.Timer.ActivePlace)}")
    
    # Cancel one timer using selective consumption
    robot.send_message_to_place(
        RobotSystem.CommandPlace,
        CommandToken("cancel_timer", {"timer_id": "task1"})
    )
    robot.tick()
    print(f"Active timers after cancel: {robot.context.token_count(RobotSystem.Timer.ActivePlace)}")
    print(f"Cancelled timers: {robot.context.token_count(RobotSystem.Timer.CancelledPlace)}")
    
    print("\n=== Testing Network Operations ===")
    
    # Send network message
    robot.send_message_to_place(
        RobotSystem.CommandPlace,
        CommandToken("send_message", {
            "destination": "192.168.1.100",
            "message": "Hello from robot!"
        })
    )
    robot.tick()
    print(f"Sent packets: {network_manager.sent_packets}")
    
    print("\n=== System Status ===")
    print(f"Status messages: {robot.context.token_count(RobotSystem.StatusPlace)}")
    
    # Show Mermaid diagram
    print("\n=== Generated Network Diagram ===")
    print(robot.generate_mermaid_diagram())


if __name__ == "__main__":
    demonstrate_system()