import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'

export default function TSConfig() {
  const location = useLocation()
  return <Navigate to="/ts/tasks?drawer=create" replace state={location.state} />
}
