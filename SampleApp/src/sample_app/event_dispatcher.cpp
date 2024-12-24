#include "event_dispatcher.hpp"

namespace SampleApp
{
	std::size_t EventHash::operator()(const Event& event) const
	{
		return std::hash<std::string>{}(event.signal);
	}

	bool Event::operator==(const Event& other) const
	{
		return signal == other.signal;
	}
	void Listener::invoke(const std::any& data) const
	{
		callback(data);
	}
	void EventDispatcher::registerListener(const Event& event, const Listener& listener)
	{
		m_listeners[event].emplace_back(listener);
	}
}