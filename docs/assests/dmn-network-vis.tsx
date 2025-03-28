import React, { useState, useEffect, useRef } from 'react';

const DMNVisualizer = () => {
  const [nodes, setNodes] = useState([]);
  const [connections, setConnections] = useState([]);
  const animationRef = useRef();
  const nodesRef = useRef([]);
  const connectionsRef = useRef([]);

  // Physics parameters
  const SPRING_K = 0.02;
  const REPEL_K = 200;
  const DAMPING = 0.95;
  const PRUNE_THRESHOLD = 0.2;

  // Initialize network
  useEffect(() => {
    const initialNodes = Array.from({ length: 5 }, (_, i) => ({
      id: i,
      x: Math.random() * 600 + 100,
      y: Math.random() * 400 + 100,
      vx: 0,
      vy: 0,
      weight: 1,
      connections: new Set()
    }));

    const initialConnections = [
      { source: 0, target: 1, weight: 1 },
      { source: 1, target: 2, weight: 1 },
      { source: 2, target: 3, weight: 1 }
    ];

    initialConnections.forEach(conn => {
      initialNodes[conn.source].connections.add(conn.target);
      initialNodes[conn.target].connections.add(conn.source);
    });

    setNodes(initialNodes);
    setConnections(initialConnections);
    nodesRef.current = initialNodes;
    connectionsRef.current = initialConnections;
  }, []);

  const updatePhysics = () => {
    const currentNodes = [...nodesRef.current];
    const currentConnections = [...connectionsRef.current];
    
    // Calculate forces
    const forces = currentNodes.map(() => ({ fx: 0, fy: 0 }));
    
    // Spring forces between connected nodes
    currentConnections.forEach(conn => {
      if (!currentNodes[conn.source] || !currentNodes[conn.target]) return;
      
      const source = currentNodes[conn.source];
      const target = currentNodes[conn.target];
      
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist === 0) return;
      
      const force = SPRING_K * (dist - 100) * conn.weight;
      
      const fx = force * dx / dist;
      const fy = force * dy / dist;
      
      forces[conn.source].fx += fx;
      forces[conn.source].fy += fy;
      forces[conn.target].fx -= fx;
      forces[conn.target].fy -= fy;
    });
    
    // Repulsion between all nodes
    for (let i = 0; i < currentNodes.length; i++) {
      for (let j = i + 1; j < currentNodes.length; j++) {
        const dx = currentNodes[j].x - currentNodes[i].x;
        const dy = currentNodes[j].y - currentNodes[i].y;
        const distSq = dx * dx + dy * dy;
        if (distSq === 0) continue;
        const dist = Math.sqrt(distSq);
        
        if (dist < 200) {
          const force = -REPEL_K / distSq;
          const fx = force * dx / dist;
          const fy = force * dy / dist;
          
          forces[i].fx += fx;
          forces[i].fy += fy;
          forces[j].fx -= fx;
          forces[j].fy -= fy;
        }
      }
    }
    
    // Update positions
    currentNodes.forEach((node, i) => {
      node.vx = (node.vx + forces[i].fx) * DAMPING;
      node.vy = (node.vy + forces[i].fy) * DAMPING;
      
      node.x += node.vx;
      node.y += node.vy;
      
      // Contain within bounds
      node.x = Math.max(50, Math.min(750, node.x));
      node.y = Math.max(50, Math.min(550, node.y));
      
      // Decay weight
      node.weight *= 0.999;
    });

    nodesRef.current = currentNodes;
    setNodes([...currentNodes]);
  };

  // Animation loop
  useEffect(() => {
    let frameCount = 0;
    
    const animate = () => {
      frameCount++;
      updatePhysics();
      
      // Occasionally add new nodes and connections
      if (frameCount % 60 === 0 && Math.random() < 0.3) {
        const currentNodes = nodesRef.current;
        const currentConnections = connectionsRef.current;
        
        const newNodeId = currentNodes.length;
        const newNode = {
          id: newNodeId,
          x: Math.random() * 600 + 100,
          y: Math.random() * 400 + 100,
          vx: 0,
          vy: 0,
          weight: 1,
          connections: new Set([Math.floor(Math.random() * currentNodes.length)])
        };
        
        const newConnection = {
          source: newNodeId,
          target: Math.floor(Math.random() * currentNodes.length),
          weight: 1
        };

        nodesRef.current = [...currentNodes, newNode];
        connectionsRef.current = [...currentConnections, newConnection];
        
        setNodes(nodesRef.current);
        setConnections(connectionsRef.current);
      }
      
      // Prune weak nodes
      if (frameCount % 100 === 0) {
        const currentNodes = nodesRef.current;
        const currentConnections = connectionsRef.current;
        
        const validNodes = currentNodes.filter(node => node.weight > PRUNE_THRESHOLD);
        const validConnections = currentConnections.filter(conn => 
          currentNodes[conn.source]?.weight > PRUNE_THRESHOLD &&
          currentNodes[conn.target]?.weight > PRUNE_THRESHOLD
        );
        
        nodesRef.current = validNodes;
        connectionsRef.current = validConnections;
        
        setNodes(validNodes);
        setConnections(validConnections);
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationRef.current);
  }, []);

  return (
    <div className="w-full h-96 bg-gray-900">
      <svg className="w-full h-full" viewBox="0 0 800 600">
        {/* Connections */}
        {connectionsRef.current.map((conn, i) => {
          const source = nodesRef.current[conn.source];
          const target = nodesRef.current[conn.target];
          if (!source || !target) return null;
          return (
            <line
              key={`conn-${i}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={`rgba(100, 200, 255, ${conn.weight * 0.5})`}
              strokeWidth={conn.weight * 2}
            />
          );
        })}
        
        {/* Nodes */}
        {nodesRef.current.map((node, i) => (
          <g key={`node-${i}`}>
            <circle
              cx={node.x}
              cy={node.y}
              r={10 + node.weight * 5}
              fill={`rgb(${150 + node.weight * 100}, 150, 255)`}
              opacity={0.7}
            >
              <animate
                attributeName="r"
                values={`${10 + node.weight * 5};${12 + node.weight * 5};${10 + node.weight * 5}`}
                dur="2s"
                repeatCount="indefinite"
              />
            </circle>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default DMNVisualizer;