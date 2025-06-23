"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import axios from "axios";

export default function TestPage() {
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const testConnection = async () => {
    setLoading(true);
    setResult("");
    
    try {
      console.log("Testing connection to http://localhost:8000/api/health");
      
      const response = await axios.get("http://localhost:8000/api/health", {
        timeout: 5000,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      console.log("Response:", response);
      setResult(`✅ Success: ${JSON.stringify(response.data, null, 2)}`);
    } catch (error) {
      console.error("Error:", error);
      if (axios.isAxiosError(error)) {
        setResult(`❌ Axios Error: ${error.message}\nCode: ${error.code}\nStatus: ${error.response?.status}`);
      } else {
        setResult(`❌ Unknown Error: ${error}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-2xl">
      <Card>
        <CardHeader>
          <CardTitle>Backend Connection Test</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button onClick={testConnection} disabled={loading}>
            {loading ? "Testing..." : "Test Connection"}
          </Button>
          
          {result && (
            <div className="p-4 bg-gray-100 rounded-md">
              <pre className="whitespace-pre-wrap text-sm">{result}</pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
