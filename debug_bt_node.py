#!/usr/bin/env python3
"""
Debug script para diagnosticar el behavior tree node.
Ejecuta: python3 debug_bt_node.py
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class BTDebugMonitor(Node):
    def __init__(self):
        super().__init__('bt_debug_monitor')
        self.get_logger().info('🔍 Iniciando monitor de diagnóstico BT...')
        
        # Verificar si el nodo principal existe
        self.get_logger().info('\n=== NODOS DISPONIBLES ===')
        self.get_logger().info(f'Buscando: /qcar2_behavior_tree_manager')
        
        # Suscribirse a los tópicos clave
        self.create_subscription(String, '/bt/state', self._on_bt_state, 10)
        self.create_subscription(String, '/bt/goal', self._on_bt_goal, 10)
        self.create_subscription(String, '/bt/mode_hybrid', self._on_bt_mode, 10)
        
        # Timer para checkear periódicamente
        self.state_received = False
        self.goal_received = False
        self.mode_received = False
        
        self.create_timer(3.0, self._check_status)
        self.get_logger().info('✓ Esperando mensajes por 10 segundos...')

    def _on_bt_state(self, msg):
        self.state_received = True
        self.get_logger().info(f'📊 /bt/state: {msg.data}')

    def _on_bt_goal(self, msg):
        self.goal_received = True
        self.get_logger().info(f'📍 /bt/goal: x={msg.data}')  # Simplificado

    def _on_bt_mode(self, msg):
        self.mode_received = True
        self.get_logger().info(f'🎛  /bt/mode_hybrid: {msg.data}')

    def _check_status(self):
        self.get_logger().warning('\n=== ESTADO DE TÓPICOS ===')
        self.get_logger().warning(f'/bt/state:        {"✓ RECIBIENDO" if self.state_received else "✗ SILENCIOSO"}')
        self.get_logger().warning(f'/bt/goal:         {"✓ RECIBIENDO" if self.goal_received else "✗ SILENCIOSO"}')
        self.get_logger().warning(f'/bt/mode_hybrid:  {"✓ RECIBIENDO" if self.mode_received else "✗ SILENCIOSO"}')
        self.get_logger().warning('\nEjecutando: ros2 node list (para verificar si existe /qcar2_behavior_tree_manager)')


def main(args=None):
    rclpy.init(args=args)
    monitor = BTDebugMonitor()
    try:
        rclpy.spin_once(monitor, timeout_sec=10.0)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
