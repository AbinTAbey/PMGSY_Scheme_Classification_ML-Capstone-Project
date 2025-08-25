"import React, { useState, useEffect, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart, Scatter, LineChart, Line } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const PMGSYDashboard = () => {
  // Sample data generation (replace with your actual CSV data)
  const generateSampleData = () => {
    const states = ['Maharashtra', 'Uttar Pradesh', 'Rajasthan', 'Madhya Pradesh', 'Bihar', 'West Bengal', 'Gujarat', 'Karnataka', 'Andhra Pradesh', 'Tamil Nadu'];
    const districts = ['District A', 'District B', 'District C', 'District D', 'District E'];
    const schemes = ['PMGSY-I', 'PMGSY-II', 'PMGSY-III', 'RCPLWEA'];
    
    const data = [];
    for (let i = 0; i < 200; i++) {
      const sanctioned = Math.floor(Math.random() * 100) + 10;
      const completed = Math.floor(Math.random() * sanctioned);
      data.push({
        id: i + 1,
        STATE_NAME: states[Math.floor(Math.random() * states.length)],
        DISTRICT_NAME: districts[Math.floor(Math.random() * districts.length)],
        PMGSY_SCHEME: schemes[Math.floor(Math.random() * schemes.length)],
        NO_OF_ROAD_WORK_SANCTIONED: sanctioned,
        NO_OF_ROAD_WORKS_COMPLETED: completed,
        LENGTH_OF_ROAD_WORK_SANCTIONED: Math.floor(Math.random() * 500) + 50,
        COST_OF_WORKS_SANCTIONED: Math.floor(Math.random() * 1000) + 100,
      });
    }
    return data;
  };

  const [data, setData] = useState([]);
  const [selectedStates, setSelectedStates] = useState([]);
  const [selectedSchemes, setSelectedSchemes] = useState([]);

  useEffect(() => {
    setData(generateSampleData());
  }, []);

  // Filter data based on selections
  const filteredData = useMemo(() => {
    let filtered = data;
    if (selectedStates.length > 0) {
      filtered = filtered.filter(item => selectedStates.includes(item.STATE_NAME));
    }
    if (selectedSchemes.length > 0) {
      filtered = filtered.filter(item => selectedSchemes.includes(item.PMGSY_SCHEME));
    }
    return filtered;
  }, [data, selectedStates, selectedSchemes]);

  // Calculate KPIs
  const kpis = useMemo(() => {
    const totalRoads = filteredData.reduce((sum, item) => sum + item.NO_OF_ROAD_WORK_SANCTIONED, 0);
    const totalCompleted = filteredData.reduce((sum, item) => sum + item.NO_OF_ROAD_WORKS_COMPLETED, 0);
    const totalLength = filteredData.reduce((sum, item) => sum + item.LENGTH_OF_ROAD_WORK_SANCTIONED, 0);
    const totalCost = filteredData.reduce((sum, item) => sum + item.COST_OF_WORKS_SANCTIONED, 0);
    const completionRate = totalRoads > 0 ? (totalCompleted / totalRoads * 100) : 0;

    return {
      totalRoads,
      totalLength,
      totalCost,
      completionRate
    };
  }, [filteredData]);

  // Prepare data for charts
  const stateWiseData = useMemo(() => {
    const grouped = {};
    filteredData.forEach(item => {
      if (!grouped[item.STATE_NAME]) {
        grouped[item.STATE_NAME] = {
          state: item.STATE_NAME,
          sanctioned: 0,
          completed: 0,
          length: 0,
          cost: 0
        };
      }
      grouped[item.STATE_NAME].sanctioned += item.NO_OF_ROAD_WORK_SANCTIONED;
      grouped[item.STATE_NAME].completed += item.NO_OF_ROAD_WORKS_COMPLETED;
      grouped[item.STATE_NAME].length += item.LENGTH_OF_ROAD_WORK_SANCTIONED;
      grouped[item.STATE_NAME].cost += item.COST_OF_WORKS_SANCTIONED;
    });
    return Object.values(grouped).sort((a, b) => b.sanctioned - a.sanctioned).slice(0, 10);
  }, [filteredData]);

  const schemeWiseData = useMemo(() => {
    const grouped = {};
    filteredData.forEach(item => {
      if (!grouped[item.PMGSY_SCHEME]) {
        grouped[item.PMGSY_SCHEME] = 0;
      }
      grouped[item.PMGSY_SCHEME] += item.NO_OF_ROAD_WORK_SANCTIONED;
    });
    return Object.entries(grouped).map(([scheme, count]) => ({
      scheme,
      count
    }));
  }, [filteredData]);

  const scatterData = useMemo(() => {
    return filteredData.map(item => ({
      ...item,
      completionRate: (item.NO_OF_ROAD_WORKS_COMPLETED / item.NO_OF_ROAD_WORK_SANCTIONED) * 100
    }));
  }, [filteredData]);

  const completionRateDistribution = useMemo(() => {
    const ranges = { '0-20%': 0, '21-40%': 0, '41-60%': 0, '61-80%': 0, '81-100%': 0 };
    filteredData.forEach(item => {
      const rate = (item.NO_OF_ROAD_WORKS_COMPLETED / item.NO_OF_ROAD_WORK_SANCTIONED) * 100;
      if (rate <= 20) ranges['0-20%']++;
      else if (rate <= 40) ranges['21-40%']++;
      else if (rate <= 60) ranges['41-60%']++;
      else if (rate <= 80) ranges['61-80%']++;
      else ranges['81-100%']++;
    });
    return Object.entries(ranges).map(([range, count]) => ({ range, count }));
  }, [filteredData]);

  const uniqueStates = [...new Set(data.map(item => item.STATE_NAME))];
  const uniqueSchemes = [...new Set(data.map(item => item.PMGSY_SCHEME))];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">PMGSY Roads Dashboard</h1>
        
        {/* Filters */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Filters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Select States:</label>
                <div className="max-h-32 overflow-y-auto border rounded p-2">
                  {uniqueStates.map(state => (
                    <label key={state} className="flex items-center mb-1">
                      <input
                        type="checkbox"
                        checked={selectedStates.includes(state)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedStates([...selectedStates, state]);
                          } else {
                            setSelectedStates(selectedStates.filter(s => s !== state));
                          }
                        }}
                        className="mr-2"
                      />
                      <span className="text-sm">{state}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Select Schemes:</label>
                <div className="max-h-32 overflow-y-auto border rounded p-2">
                  {uniqueSchemes.map(scheme => (
                    <label key={scheme} className="flex items-center mb-1">
                      <input
                        type="checkbox"
                        checked={selectedSchemes.includes(scheme)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedSchemes([...selectedSchemes, scheme]);
                          } else {
                            setSelectedSchemes(selectedSchemes.filter(s => s !== scheme));
                          }
                        }}
                        className="mr-2"
                      />
                      <span className="text-sm">{scheme}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-4 flex gap-2">
              <button 
                onClick={() => { setSelectedStates([]); setSelectedSchemes([]); }}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Clear All Filters
              </button>
            </div>
          </CardContent>
        </Card>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-6">
              <div className="text-2xl font-bold text-blue-600">{kpis.totalRoads.toLocaleString()}</div>
              <p className="text-gray-600">Total Roads Sanctioned</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="text-2xl font-bold text-green-600">{kpis.totalLength.toLocaleString()}</div>
              <p className="text-gray-600">Total Length (km)</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="text-2xl font-bold text-purple-600">₹{kpis.totalCost.toLocaleString()}</div>
              <p className="text-gray-600">Total Cost (Crores)</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="text-2xl font-bold text-orange-600">{kpis.completionRate.toFixed(1)}%</div>
              <p className="text-gray-600">Completion Rate</p>
            </CardContent>
          </Card>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* State-wise Progress */}
          <Card>
            <CardHeader>
              <CardTitle>Top 10 States: Roads Sanctioned vs Completed</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stateWiseData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="state" 
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    fontSize={12}
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="sanctioned" fill="#8884d8" name="Sanctioned" />
                  <Bar dataKey="completed" fill="#82ca9d" name="Completed" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Scheme Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Distribution by PMGSY Scheme</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={schemeWiseData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ scheme, count, percent }) => `${scheme}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {schemeWiseData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Cost vs Length Analysis */}
          <Card>
            <CardHeader>
              <CardTitle>Cost vs Length Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={scatterData.slice(0, 50)}>
                  <CartesianGrid />
                  <XAxis 
                    type="number" 
                    dataKey="LENGTH_OF_ROAD_WORK_SANCTIONED" 
                    name="Length (km)"
                    label={{ value: 'Length (km)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="COST_OF_WORKS_SANCTIONED" 
                    name="Cost (Crores)"
                    label={{ value: 'Cost (Crores)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-2 border rounded shadow">
                            <p>{`State: ${data.STATE_NAME}`}</p>
                            <p>{`Length: ${data.LENGTH_OF_ROAD_WORK_SANCTIONED} km`}</p>
                            <p>{`Cost: ₹${data.COST_OF_WORKS_SANCTIONED} Cr`}</p>
                            <p>{`Scheme: ${data.PMGSY_SCHEME}`}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter dataKey="COST_OF_WORKS_SANCTIONED" fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Completion Rate Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Distribution of Completion Rates</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={completionRateDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Data Summary Table */}
        <Card>
          <CardHeader>
            <CardTitle>Data Summary by State</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="min-w-full table-auto">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-4 py-2 text-left">State</th>
                    <th className="px-4 py-2 text-left">Roads Sanctioned</th>
                    <th className="px-4 py-2 text-left">Roads Completed</th>
                    <th className="px-4 py-2 text-left">Total Length (km)</th>
                    <th className="px-4 py-2 text-left">Total Cost (Cr)</th>
                    <th className="px-4 py-2 text-left">Completion Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {stateWiseData.slice(0, 10).map((item, index) => (
                    <tr key={index} className={index % 2 === 0 ? "bg-gray-50" : "bg-white"}>
                      <td className="px-4 py-2">{item.state}</td>
                      <td className="px-4 py-2">{item.sanctioned.toLocaleString()}</td>
                      <td className="px-4 py-2">{item.completed.toLocaleString()}</td>
                      <td className="px-4 py-2">{item.length.toLocaleString()}</td>
                      <td className="px-4 py-2">₹{item.cost.toLocaleString()}</td>
                      <td className="px-4 py-2">{((item.completed / item.sanctioned) * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PMGSYDashboard;
