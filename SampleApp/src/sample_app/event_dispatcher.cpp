#include "event_dispatcher.hpp"

namespace SampleApp
{
	std::size_t EventHash::operator()(const Event& event_) const
	{
		return std::hash<std::string>{}(event_.signal);
	}

	bool Event::operator==(const Event& other) const
	{
		return signal == other.signal;
	}
	void Listener::invoke(const std::any& data) const
	{
		callback(data);
	}
	void EventDispatcher::registerListener(const Event& event_, const Listener& listener)
	{
		if (m_listeners.find(event_) == m_listeners.end())
		{
			m_listeners[event_] = std::vector<Listener>();
		}
		m_listeners[event_].emplace_back(listener);
	}
}