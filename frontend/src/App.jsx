import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatPage from "./pages/ChatPage";

export default function App() {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [temperature, setTemperature] = useState(0.5);
  const [hasDatasets, setHasDatasets] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar
        selectedDataset={selectedDataset}
        onSelectDataset={setSelectedDataset}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        onDatasetsChange={(list) => setHasDatasets(list.length > 0)}
      />
      <ChatPage datasetId={selectedDataset} temperature={temperature} hasDatasets={hasDatasets} />
    </div>
  );
}
